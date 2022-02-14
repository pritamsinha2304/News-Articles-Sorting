# News Articles Sorting

## Introduction

Every day millions of news articles got created and broadcast over the internet about everything in the world ranging from wellness, parenting to environment, disasters etc. This is how we the people keeps updated ourselves about the news from around the world. News are just data and every day millions of these data are created and got tagged by various sources about its content. Based on these tags, people are attracted towards various news. But manually tagging or sorting those news articles worldwide is a tedious job, since it involves going through news articles to tag them to a genre. Since the advent of AI, many techniques were invented and one of these techniques deals with text data. This work discusses the implementation of that technique, called Natural Language Processing (NLP) to train an AI to classify news article based on their content and tag them with a genre.

This web app is built on the basis on the title and to predict news articles on user's input. The page is divided into 3 sections, which is based on user's input preference. Each section is well-defined and users can give inputs. Each section on predicting completely and successfully, will go to the metrics page, where all the metrics of the prediction results are shown.


## Input Data Selection Options

The web app input options is divided into 3 sections/forms. Each section deals with specific types of user inputs and can handles only that file format. The idea is to seperate the form based ont he quantity of users's input.
1. Single Data: This is used only when the user have single article to predict.
2. Multiple/Batch Data: This is used when the user possess multiple/batch of articles he/she wants to predict. Only csv file format is supported here.
3. Demo Data: This is used in case the user doesn't have any data. The user just to look around the app and see what the app has to offer.

<img src="/app images/homepage.png"/>



### Single Data

The single data form consists of 3 fields. 
<ul>
  <li>One is field for articles, which is required.</li> 
  <li>Second is model selection, which model to use to predict, currently only one is there.</li>
  <li>And the third the actual category/label of the articles uploaded. The actual label is optional though. But if you have, select the actual label from dropdown option of 28    available categories.</li>
</ul>

<img src="/app images/single.png"/>



### Multiple/Batch Data

If you have batches to articles to predict, use this. It consists of 3 fields-
<ul>
  <li>Upload the file containing all the articles in .csv format. Currently only csv file format is supported. The csv file must have single column of name "Text".</li>
  <li>Second is model selection, which model to use to predict, currently only one is there.</li>
  <li>Upload the ffile for actual label, if you have. The file must be csv file with single column of name "Category".</li>
</ul>

<img src="/app images/multiple.png"/>



### Demo Data

You don't have any data, just wanna look around.
Select the model and you are good to go.

<img src="/app images/demo.png"/>



## Metrics page

The metrics shows all the metrics related to the classification problem. The uploaded data from the user after predicting will be shown in the metrics page. The metrics are possible to show only when the user uploaded actual label of the data. The results will be shown in tabulated format. If the user didn't uploaded actual data, only the prediction of the uploaded data will be shown.

<img src="/app images/metrics.png"/>
