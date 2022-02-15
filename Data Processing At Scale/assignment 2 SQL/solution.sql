CREATE TABLE query1 AS
SELECT g.name AS name, COUNT(DISTINCT(h.movieid)) AS moviecount
FROM hasagenre h, genres g
WHERE h.genreid=g.genreid
GROUP BY g.genreid;

CREATE TABLE query2 AS
SELECT g.name AS name, avg(r.rating) AS rating
FROM genres g, hasagenre h, ratings r
WHERE g.genreid=h.genreid AND h.movieid=r.movieid
GROUP BY g.genreid;

CREATE TABLE query3 AS
SELECT m.title as title, count(r.rating) as countofratings
FROM movies m, ratings r
WHERE m.movieid=r.movieid
GROUP BY m.movieid
HAVING count(r.rating) >= 10;

CREATE TABLE query4 AS
SELECT m.movieid AS movieid, m.title AS title
FROM movies m, hasagenre h, genres g 
WHERE m.movieid=h.movieid AND h.genreid=g.genreid
AND g.name='Comedy';

CREATE TABLE query5 AS
SELECT m.title AS title, AVG(r.rating) AS average
FROM movies m, ratings r
WHERE m.movieid=r.movieid
GROUP BY m.movieid;

CREATE TABLE query6 AS
SELECT AVG(r.rating) AS average
FROM ratings r, movies m, hasagenre h, genres g
WHERE r.movieid=m.movieid AND m.movieid= h.movieid AND h.genreid = g.genreid
AND g.name='Comedy';

CREATE TABLE query7 AS
SELECT AVG(r.rating) AS average
FROM ratings r, movies m, hasagenre h, genres g
WHERE r.movieid=m.movieid AND m.movieid= h.movieid AND h.genreid = g.genreid
AND g.name='Comedy' AND m.movieid IN 
 (SELECT m.movieid
  FROM movies m, hasagenre h, genres g 
  WHERE m.movieid= h.movieid AND h.genreid = g.genreid AND g.name='Romance');
  
CREATE TABLE query8 AS
SELECT AVG(r.rating) AS average
FROM ratings r, movies m, hasagenre h, genres g
WHERE r.movieid=m.movieid AND m.movieid= h.movieid AND h.genreid = g.genreid
AND g.name='Romance' AND m.movieid NOT IN 
 (SELECT m.movieid
  FROM movies m, hasagenre h, genres g 
  WHERE m.movieid= h.movieid AND h.genreid = g.genreid AND g.name='Comedy');
 
CREATE TABLE query9 AS
SELECT m.movieid AS movieid, r.rating as rating 
FROM movies m, ratings r
WHERE  m.movieid = r.movieid AND r.userid = :v1; 

CREATE TABLE ratedmovies AS
(SELECT m.movieid AS movieid, m.title AS title, r.rating AS rating
FROM ratings r, movies m
WHERE r.userid = :v1 AND r.movieid = m.movieid);

CREATE TABLE unratedmovies AS
SELECT m.movieid, m.title
FROM movies m 
WHERE m.movieid NOT IN 
        (SELECT rm.movieid 
		 FROM ratedmovies rm);

CREATE TABLE similarity AS
SELECT  mov1.movieid AS movieid_1, mov2.movieid AS movieid_2, mov2.rating AS rating, (1 - ABS(mov1.average - mov2.average)/5) AS sim
FROM 
(SELECT * FROM unratedmovies m1, query5 q5 WHERE m1.title = q5.title ) AS mov1,
(SELECT * FROM ratedmovies m2, query5 q5 WHERE m2.title = q5.title ) AS mov2
WHERE mov1.movieid != mov2.movieid;

CREATE TABLE recommendation AS
SELECT m.title AS title
FROM movies m
WHERE m.movieid IN
(SELECT s.movieid_1
FROM similarity s
GROUP BY s.movieid_1
HAVING SUM(s.sim * s.rating)/SUM(s.sim) > 3.9);