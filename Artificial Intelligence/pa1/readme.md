### CSE 571: Course Name

# Bayesian Networks

## Purpose:

Students will create Bayesian Networks and use them to perform inferences. Students will gain
hands-on experience with BN inferences. This project provides an excellent opportunity to gain
further exposure to relevant topics and applications (including problem diagnosis and monitoring
and filtering). The successful completion of this project will challenge students to develop
applications that use Bayesian Networks.

## Objectives:

Students will be able to:
● Create Bayesian Networks.
● Determine inferences from Bayesian Networks.
● Model real-world problems in Bayesian Networks.

## Technology Requirements:

```
● Linux (windows user may install virtual machines)
● Python 3.4 or higher
● Download and install pip and then install pgmpy:
○ $ git clone https://github.com/pgmpy/pgmpy
○ $ cd pgmpy/
○ $ sudo pip install -r requirements.txt
○ $ sudo python setup.py install
*Note ​: if you encountered problems installing pip or pgmpy, refer to the ​pgmpy
Installation Page​.
**You can find the documents for pgmpy on the ​pgmpy Documentation Page​.
```
## Project Description:


1. Familiarize with the Bayesian Model (BN) class in pgmpy library. An example (​ **bn.py** ​)
   illustrating BN construction an inference is provided for the following BN:
   Run bn.py by “python bn.py”. The following shows you the results of two queries:
   a. P(D|-c) = {0.65, 0.35}
   b. P(C|-s, -p) = {0.97, 0.03}
2. Answer the following questions using the provided code ​ **and** ​ by hand to see whether
   they match (this question is not graded):
   a. P(+d|+s)
   b. P(+x|+d,-s)
   c. Does pgmpy return exact results (up to the system’s accuracy)?
3. Create code for the following BN:


Artificial Intelligence: A Modern Approach​ _3rd Edition_ ​.
Save it as “​ **burglary.py** ​”. Important: please follow the instructions in the template provided to
you to name your variables and structure your code.

4. Answer the following questions using your code ​ **and** ​ by hand to see whether they match
   (this question is not graded but the code output should match with your computation by
   hand):
   a. P(+j|-e)
   b. P(+m|+b,-e)
   c. P(+m|+b,+e)
   d. P(+m|+j)
   e. P(+m|+j,-b,-e)
5. Familiarize with the Dynamic Bayesian Model (DBN) class in pgmpy library. An example
   (​ **dbn.py** ​) illustrating DBN construction an inference is provided for the following DBN:


Run dbn.py by “python dbn.py”. The following shows you the results of a query:
a. P(G3|g0=1, g1=2) = {0.4358, 0.2552, 0.3090}​ ​ (the distribution of G at the 3rd
time slice given g0=1 at the zeroth step and g1=2 at the first step)

6. Create code for the DBN for the following problem, which is similar to the problem
   discussed in our DBN lecture:


a. You agent always move in a clockwise fashion
b. When it moves, it has a 50% chance of moving to the desired location and 50% it
stays where it was.
c. The robot is equipped with a sensor that returns the correct position with a 60%
chance and a random position (including the correct position) with a 40% chance
d. The agent starts at C at time 0.
Save it as “​ **agent.py** ​”. Important: please follow the instructions in the template provided
to you to name your variables and structure your code.

7. Test your code thoroughly.
   For example, P(Location1 = A | Sensor 1=C)= 0.125 (The probability of the agent at
   location A at step 1 given that the sensor at step 1 returns C.

## Submission Directions for Project Deliverables

Submission templates and resources for this project can be found in the “CSE 571_Bayesian
Networks_Project Templates and Resources” for you to download.
**Files to submit:**
● burglary.py
● agent.py


