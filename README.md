# Question Answering system on nptel video

This repository create QA system on nptel video. It takes subtitle, question and the options as the input and predict the correct output.
 [This](https://www.youtube.com/playlist?list=PLYihddLF-CgYuWNL55Wg8ALkm6u8U7gps) NPTEL video lacture series was used as the dataset. It's questions were created by myself.
 Around 450 questions were generated. First 30 videos are included in the dataset and there is around 2 questions per 5 minutes.
 Subtitle file is very huge hence it is devided in the timestamp of 5 minutes. Similarly QA file also devided in the same timespamp of 5 minute.
 
 Subtitle will pass through the lstm layer, question will pass through another lstm layer and 4 options will pass through 4 different lstm layers
 after that they all will be concatenated and pass through couple of dence layers and at the end softmax layer to find out which of the 4 options
 is the answer of our question? Then we will compare it with the target answer and train our model.
 
 At max 50% of accurecy was achieved because of the lack of data more accurecy can be achieved if more data is available. There is also the problem of overfitting.
 Which was solved upto some amount by early stopping and drop out.
 
 To see the output run the file final.py by the command 
 
 python final.py
