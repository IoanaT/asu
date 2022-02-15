package cse512

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._

import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._

import scala.collection.immutable.ListMap

object HotcellAnalysis {
  Logger.getLogger("org.spark_project").setLevel(Level.WARN)
  Logger.getLogger("org.apache").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)
  Logger.getLogger("com").setLevel(Level.WARN)


  def calcMean(xj:Long, numcells:Double):Double = {
    return xj/numcells
  }

  def calcStdDev(xjsquare:Long, numcells:Double, mean:Double):Double = {
    val stdDev = Math.sqrt(xjsquare/numcells - mean*mean)
    return stdDev
  }

  def calcGetisOrd(wijxj:Int, mean:Double, stddev:Double, numCells:Int):Double = {
    val num = wijxj - mean*27
    val den = stddev*math.sqrt( (numCells*27 - (27*27)  )/ numCells-1 )
    if (den == 0.0){
      return 0.0
    }
    return num/den
  }

  def runHotcellAnalysis(spark: SparkSession, pointPath: String): DataFrame =
  {
    // Load the original data from a data source
    var pickupInfo = spark.read.format("com.databricks.spark.csv").option("delimiter",";").option("header","false").load(pointPath);
    pickupInfo.createOrReplaceTempView("nyctaxitrips")
    pickupInfo.show()

    // Assign cell coordinates based on pickup points
    spark.udf.register("CalculateX",(pickupPoint: String)=>((
      HotcellUtils.CalculateCoordinate(pickupPoint, 0)
      )))
    spark.udf.register("CalculateY",(pickupPoint: String)=>((
      HotcellUtils.CalculateCoordinate(pickupPoint, 1)
      )))
    spark.udf.register("CalculateZ",(pickupTime: String)=>((
      HotcellUtils.CalculateCoordinate(pickupTime, 2)
      )))
    pickupInfo = spark.sql("select CalculateX(nyctaxitrips._c5),CalculateY(nyctaxitrips._c5), CalculateZ(nyctaxitrips._c1) from nyctaxitrips")
    var newCoordinateName = Seq("x", "y", "z")
    pickupInfo = pickupInfo.toDF(newCoordinateName:_*)
    pickupInfo.show()

    // Define the min and max of x, y, z
    val minX = -74.50/HotcellUtils.coordinateStep
    val maxX = -73.70/HotcellUtils.coordinateStep
    val minY = 40.50/HotcellUtils.coordinateStep
    val maxY = 40.90/HotcellUtils.coordinateStep
    val minZ = 1
    val maxZ = 31
    val numCells = (maxX - minX + 1)*(maxY - minY + 1)*(maxZ - minZ + 1)

    // YOU NEED TO CHANGE THIS PART

    // return pickupInfo // YOU NEED TO CHANGE THIS PART

    // 1) retrieve pickup points for all cells within the time-space cube
    pickupInfo.createOrReplaceTempView("pickupInfo")
    var pickupPoints = spark.sql(s"select x,y,z,count(*) as xi, count(*) * count(*) as xisquare FROM pickupInfo WHERE z <= $maxZ AND z >= $minZ AND y <= $maxY AND y >= $minY AND x <= $maxX AND x >= $minX group by x,y,z")
    pickupPoints.createOrReplaceTempView("pickupPoints")
    //print("pickupPoints")
    //pickupPoints.show()

    pickupInfo.createOrReplaceTempView("pickupWIJXJ")
    pickupInfo = spark.sql(s"select p.x,p.y,p.z,sum(n.xi) AS wijxj FROM pickupPoints as p, pickupPoints as n WHERE n.z <= p.z+1 AND n.z >= p.z-1 AND n.y <= p.y+1 AND n.y >= p.y-1 AND n.x <= p.x+1 AND n.x >= p.x-1 group by p.x,p.y,p.z")
    pickupInfo.createOrReplaceTempView("pickupWIJXJ")
    //println("wijxj")
    //pickupInfo.show()

    val xj =  pickupPoints.agg(sum("xi")).first().getLong(0)
    val xjsquare = pickupPoints.agg(sum("xisquare")).first().getLong(0)

    val mean = calcMean(xj, numCells)
    val stdDev = calcStdDev(xjsquare, numCells, mean)

    spark.udf.register("calcGetisOrd", (wijxj:Int, xj:Int) => (calcGetisOrd(wijxj, mean, stdDev, numCells.toInt)) )
    var zscore = spark.sql("select a.x,a.y,a.z,b.wijxj,a.xi, calcGetisOrd(b.wijxj, a.xi) as getisOrd from pickupPoints a, pickupWIJXJ b where a.x=b.x AND a.y=b.y AND a.z=b.z order by getisOrd desc limit 200")
    zscore.createOrReplaceTempView("zscore")

    //print("zscore")
    //zscore.show(200)

    var result = zscore.drop("wijxj","xi","getisOrd")
    return result

  }
}







