# Project #2 Instructions: Analyzing Code Coverage Instructions

## **Introduction:**

Project 2 will span Units 4 and 5 and is due by **11:59 PM PST February 16th** so plan accordingly. Project 2 requires you to use structural based testing techniques to develop test cases for control flow analysis and identify data flow anomalies in the given code. A PDF of the assignment details can be found below for your convenience.

## **Part 1:**

Select any tool that provides statement and decision code coverage. Utilizing the VendingMachine.java code given to you, develop a set of test cases for your code based on the following requirements:

-   Takes in an integer input

-   Allows users to select between three products: Candy (20 cents), Coke (25 cents), Coffee (45 cents)

-   Returns the selected product and any remaining change

-   If there is not enough money to buy the product, displays the amount necessary to buy the product and other products to purchase.


Execute the program with your test cases and observe the code coverage of your test cases. The goal is to reach 100% in statement coverage and at least 90% decision coverage. For decision coverage, please make sure to test all the decisions except the False decision for line 32 (input < 45).

## **Please submit the following:**

1.  Description of the tool used and the types of coverage it provides

2.  Set of test cases

3.  Screenshot showing the coverage achieved for the test cases developer

4.  Your evaluation of the tool's usefulness


## **Part 2:**

Select any static source code analysis tool. The StaticAnalysis.java code given to you contains two different data flow anomalies. Execute the tool on StaticAnalysis.java and identify what the two data flow anomalies are. The inputs are:

-   the weight of the package as an integer

-   the length of the package as an integer

-   the type of product as a String


## **Please submit the following:**

1.  Description of the tool used and the types of analysis it provides

2.  Description of the two data flow anomalies

3.  Screenshot showing the analysis performed

4.  Your evaluation of the tool’s usefulness