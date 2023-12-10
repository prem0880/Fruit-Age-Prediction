# Fruit-Age-Prediction

Summary
- Fruit age estimation using CNN. On providing an image of a fruit, the model will output the estimated number of days left for that fruit to reach decay state.

Detail
- A image of a fruit is captured and uploaded through mobile app which is developed using java into firebase database.
- Then the uploaded image is retrived and given as input to the model.The model predicts the freshness and age of the fruit(in days).
- Using this the freshness of the fruit is calculated(in %) and the uploaded into firebase.Then, the mobile app fetches and displays the result.
