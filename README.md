# Phishing-URL-detection-using-GAN
Eroding Web Phishing Attack by Generated Adversarial
Network(GANS)

1. Motivation Behind the Project
Phishing attacks remain one of the most pervasive and evolving threats in the digital age. These attacks
deceive individuals into providing sensitive information by mimicking legitimate sources. Traditional
detection methods, including static rule-based systems and manual feature extraction, have proven
inadequate due to their inability to adapt to the dynamic nature of phishing strategies. The project
addresses these limitations by implementing Generative Adversarial Networks (GANs) to create a
self-improving, adaptive system for detecting phishing URLs. This innovation aligns with the growing
need for automated, real-time solutions that can tackle sophisticated cyberattacks while maintaining
high detection accuracy.
2. Type of Project
Development cum Research Project
The project involves both the development of a GAN-based phishing detection system and research
into the effectiveness of GANs compared to traditional machine learning and deep learning approaches.
3. Critical Analysis of Research Papers and Gaps in Work
Research Papers Analyzed:
● Sasi et al. (2024): This paper proposes a GAN-based phishing URL detection system with a
VAE generator and a transformer discriminator. It achieved 97.75% accuracy on a dataset of
one million URLs, outperforming traditional models. The study highlights future potential for
real-time applications like browser plugins
● Kamran et al. (2024):This paper proposes a semi-supervised Conditional GAN for phishing
URL detection, integrating game theory to enhance security. The model achieved 95.52%
accuracy on 500,000 URLs, surpassing traditional methods, and demonstrated robustness
against adversarial examples. Fast inference and attacker-aware design make it ideal for
real-time cybersecurity applications.
● Robic-Butez et al. (2020): This paper proposes a GAN-based approach for phishing website
detection, by using Address based , Html or java script based and metadata based features. The
model achieved 94% accuracy on datasets like PhishTank, adapting well to emerging phishing
tactics. Future enhancements include adding a "suspicious" category and integrating with
broader security systems.
4. Features Built and Programming Language Used
● Features: Data processing and feature selection for selecting 10 url or address based features,
and GAN for synthetic data generation.
● Programming Language: Python.
● Libraries Used: Pandas, NumPy, TensorFlow etc.
5. Proposed Methodology
1. Dataset Collection The dataset was sourced from Kaggle, containing 11,430 URLs equally divided
into legitimate and phishing categories. Over 80 URL-based features were available for initial
exploration.
2. Feature Selection Ten essential Url or Address based features were selected for simplicity and
classification effectiveness. These features were critical in distinguishing between legitimate and
phishing URLs.
3. GAN Model Training
● Generator: Produced synthetic phishing features resembling legitimate data.
● Discriminator:trained to differentiate between real and synthetic data.
● Adversarial Training: Alternating phases for generator and discriminator ensured balanced
learning and improved performance.
4. Evaluation and Optimization Loss functions for both networks were monitored every 10 epochs to
ensure effective training and convergence.
5. Synthetic Data Generation : After training, the generator was used to create 5,715 synthetic
phishing features, which were merged with real data for further evaluation and model improvement in
future.
6. Algorithm/Description of the Work
● Dataset Preparation: Collected a dataset of 11,430 URLs, equally split between legitimate
and phishing categories.
● Feature Selection: Selected 10 critical URL-based features for simplicity and effective
classification.
● Generator Role: Configured the generator to create synthetic phishing features resembling
legitimate data.
● Discriminator Role: Set up the discriminator to classify data as either real or synthetic.
● GAN Setup: Designed the GAN model with a generator and discriminator.
● Training Loop: Alternated between training the generator and the discriminator to ensure
balanced learning.
● Hyperparameter Tuning: Used Adam optimizer with appropriate learning rates and trained
over 500 epochs.
● Loss Monitoring: Evaluated generator and discriminator loss values every 10 epochs to
ensure effective training.
● Synthetic Data Generation: Generated 5,715 synthetic phishing features that closely mimic
real phishing data.

8. Results
● Generator Output: 5,715 synthetic phishing data points closely mimicking real phishing
features.
● Discriminator Performance: Improved classification capabilities through iterative training.
● Final merged dataset with real and synthetic phishing data showed robust detection
performance.
9. Conclusion
This project successfully implemented a GAN-based approach to phishing URL detection, overcoming
the limitations of static, rule-based systems. By leveraging adversarial training, the system
demonstrated adaptability to evolving phishing tactics and improved classification accuracy. The use of
synthetic data enriched the dataset, providing a more comprehensive training environment. Despite its
success, challenges in scalability and computational requirements highlight areas for future
improvement. Potential enhancements include optimizing the architecture for real-time applications
and integrating additional feature types for broader applicability. This project sets a benchmark for
innovative phishing detection solutions, contributing significantly to the field of cybersecurity.
10. Overall design of project

![GAN](https://github.com/user-attachments/assets/a7233582-dfa6-42d5-8456-94b92ed412bd)


![1 DATA HANDLING](https://github.com/user-attachments/assets/bb2d4456-d9a1-4f44-aca5-dc9cd755cdc3)

![2 GENERATOR MODEL BUILD](https://github.com/user-attachments/assets/3e79c803-bb18-466c-964d-211f97641129)

![3  DISCRIMINATOR BUILDING](https://github.com/user-attachments/assets/d5278079-198f-43ea-bd5a-e194e2d8dc34)

![4  GAN MODEL COMPILATION](https://github.com/user-attachments/assets/867f2b06-9193-4505-931f-d11df51c1be2)


![5  TRAING GAN](https://github.com/user-attachments/assets/5fb42298-81d6-4e70-b41c-a7714823afec)

![6  SYN DATA GENERATION](https://github.com/user-attachments/assets/e86417d7-e9bd-4e2d-851d-80be490888a2)

![7  MERGED DATASET CREATION](https://github.com/user-attachments/assets/c5570720-6339-4eca-9968-b73cb5246564)

![ACCURACY CALCULATION](https://github.com/user-attachments/assets/a9fa864b-764e-4dff-b999-75dd1f079d42)



