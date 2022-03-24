/* Java program for Vending Machine. The class takes in the 2 parameters
and returns whether the item can be dispensed or not */

public class VendingMachine
{

    public static String dispenseItem(int input, String item)
    {
        int cost = 0;
        int change = 0;
        String returnValue = "";
        if (item == "candy")
            cost = 20;
        if (item == "coke")
            cost = 25;
        if (item == "coffee")
            cost = 45;

        if (input > cost)
        {
            change = input - cost;
            returnValue = "Item dispensed and change of " + Integer.toString(change) + " returned";
        }
        else if (input == cost)
        {
            change = 0;
            returnValue = "Item dispensed.";
        }
        else
        {
            change = cost - input;
            if(input < 45)
                returnValue = "Item not dispensed, missing " + Integer.toString(change) + " cents. Can purchase candy or coke.";
            if(input < 25)
                returnValue = "Item not dispensed, missing " + Integer.toString(change) + " cents. Can purchase candy.";
            if(input < 20)
                returnValue = "Item not dispensed, missing " + Integer.toString(change) + " cents. Cannot purchase item.";
        }

        return returnValue;

    }
}