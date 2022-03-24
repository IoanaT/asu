# Project #3 Part 1: Risk Based Testing

## **Introduction:**

Project 3 will span Units 6 and 7 and is split into two parts. Project 3 will cover both risk based testing and reliability prediction approaches. Project 3 Part 1 will require performing risk based analysis given a real life scenario. Both parts of Project 3 will be due at the end of Unit 7 by **11:59 PM MST March, 1st** so plan accordingly. A PDF of the assignment details can be found below for your convenience.

## **Project #3 Part 1 Details:**

You are working as a project manager for a software testing team in charge of testing a music player application. You are required to come up with an order for the completion of testing tasks before the application is deployed. There are various tasks that need to be completed. A quarter of the application was built upon legacy code that has not been documented or tested very well. Sometimes, there are software compatibility issues with the rest of the software. This code tends to crash much of the time. Unfortunately, if there is an error, it will bring down the application for multiple days, rendering the site useless. The process of testing and fixing the error would take a long time because of the lack of documentation.

Another portion of the application for the podcasts section was outsourced to a smaller development company that typically does a good job with their development. They document their code well and always ensure the application can be available even if there are small issues. Since this is a smaller portion of the application that fewer users use, there would be very minimal damage if the section was not working for some time, and it would not affect usability. There is also a very low likelihood that the page would stop working because it has been tested well.

The rest of the code was built by the developers on your partner team. They all have at least a couple years of development experience, and have properly documented any changes and additions they have made. This makes it very easy for the testers to develop tests. If parts of the new code have defects, there is a high chance that other portions of the application could not be used, which would heavily affect usability and availability of the application. Since the developers are very experienced and have worked on similar code before, there is a very low likelihood of having a significant number of errors.

The entire code itself is very complex. It calls upon many other APIs and depends on the functionality of other software developed by your company. The likelihood that an API can go down is very high, so it is vital that the application is able to handle this issue. The developers have designed the application to have the ability where if one of the APIs goes down, the application would instead take the local data it has stored for the API to keep the application running. The only downsides would be that performance would be slower and users would not get the latest information. The consequences are not as severe as if the legacy code was down, but are more severe than if the podcast section was down.

With this knowledge, it is up to you to create an order that your team will follow during testing. As a reference, minimal testing is needed if the risk exposure is low. Average testing is needed if the risk exposure medium. Maximum testing is needed if risk exposure is high.Risk exposure = P(adverse event occurring) * consequences. The following chart outlines risk exposure based on the probability and consequences.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/AQTpfnD4Eem7ixL_6m9HFg_89bf0632bee401e784025fc1e030ccb5_Project-_3.png?expiry=1648252800000&hmac=ZUEUNuyMV-5ZTOY3bp7xvSExHEBCsZ-EpDWxzr_NcZY)

## **In this project, there are three tasks:**

1.  Prioritize testing order based on risk exposure for the 4 tasks below. Rank them on the table and provide a brief explanation for your rankings. a) Test legacy code. b) Test outsourced code c) Test APIs d) Test new code

2.  Rate the intensity of testing needed from 3 categories (minimal, average, maximum)

3.  Create a contingency plan to mitigate risk from the highest risk testing task.


Project #3 Part 2 is located in Unit 7 along with the area to submit both parts of the project.