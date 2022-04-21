# COSC_490_Project
Happywhale - Whale and Dolphin Identification 
(Identify whales and dolphins by unique characteristics)

Report - https://drive.google.com/file/d/1TlwvVtV2Gy5pnuP5_sWyuzWZppk2QEnA/view?usp=sharing

![image](https://user-images.githubusercontent.com/19433397/164393590-25a84828-7002-4bbc-a4b0-8028274c488a.png)

We have been using fingerprints and facial recognition to identify people, but can we use similar approaches with animals? Researchers track marine life by the shape and markings on their tails, dorsal fins, heads and other body parts. Identification by natural markings via photographs - known as photo-ID - is a powerful tool for marine mammal science. Researchers use these photo IDs to track individual mammals over time, which in turn enables assessments of population status and trends. However, researchers mostly rely on manual processes to identify the mammals using their photo-ID. In this project, we aim to automate the process of identifying whales and dolphins using their photo-ID. This can lead to a decrease in identification times for researchers. More efficient identification could enable a scale of study previously unaffordable or impossible.

Currently, most research institutions rely on time-intensive—and sometimes inaccurate—manual matching by the human eye. A lot of time is spent in manual matching, where the researchers have to stare at multiple photos to compare one individual to another, find matches, or identify new individuals. While researchers enjoy the process, manual matching limits the scope and reach.



EXPLORATORY DATA ANALYSIS

Total training dataset size: 51033 Samples.

![image](https://user-images.githubusercontent.com/19433397/164393966-42d9974f-09bc-4506-ad25-d331b2b642e4.png)

![image](https://user-images.githubusercontent.com/19433397/164394037-8fbcae2d-fcec-4e3d-83f5-c313fbcb0bfd.png)

![image](https://user-images.githubusercontent.com/19433397/164394174-41949d4c-dceb-4796-a08b-90eec997ae1c.png)

Total Unique IDs: 15587
Number of Individuals with just one image: 9258

<img width="988" alt="image" src="https://user-images.githubusercontent.com/19433397/164405174-cc83b754-924f-43ec-9cc5-f74728058264.png">

<img width="1133" alt="image" src="https://user-images.githubusercontent.com/19433397/164405403-fb321444-6b30-428c-a9a6-ac3e53bf6b45.png">


IMPLEMENTATION: 
- ConvNeXT
- ArcMarginProduct (ArcFace) Loss Function

FRAMEWORKS USED:
PyTorch,
PyTorch Lightning,
Timm,
faiss-gpu,
Numpy,
Pandas,
Scikit-Learn
