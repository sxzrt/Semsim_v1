# Privacy Assessment on Reconstructed Images: Are Existing Evaluation Metrics Faithful to Human Perception?

This repository serves as a demonstration of the techniques outlined in the paper [Privacy Assessment on Reconstructed Images: Are Existing Evaluation Metrics Faithful to Human Perception?](https://arxiv.org/pdf/2309.13038.pdf). The full codebase will be made available by SonyAI, pending approval from their legal department.

[**Project Page**](https://sites.google.com/view/semsim)



## Getting Started
####  Step1:  Train the classifier to be evaluated and attack the classifier to obtain reconstructed images

For this initial step, the primary scripts and resources are located in the following repositories: [ATSPrivacy](https://github.com/gaow0007/ATSPrivacy), [Inverting Gradients](https://github.com/JonasGeiping/invertinggradients) and [DLG](https://github.com/mit-han-lab/dlg). These contain the necessary code to train the classifier and perform the attacks to generate reconstructed images.

To proceed, you'll need original images. There are two options:

* **use provided [CIFAR-100 samples](https://drive.google.com/file/d/1TjRNUX5KTzEAXYVhCHROD5ZVE5uFNosE/view?usp=drive_link)**

* **use your own dataset**:  integrate your dataset by modifying the dataset_dir variable in the code to point to your dataset's location.

Additionally, we offer a set of reconstructed images for this step, available for download [here](https://drive.google.com/file/d/12AXAPTTRyDfUJ3s807Oy-CxXk3E1Py9z/view?usp=sharing).

#### Step2: Evaluate privacy leakage using various metrics


* **Exisitng metric**: Mean Squared Error (MSE), Structural Similarity Index (SSIM), and Peak Signal-to-Noise Ratio (PSNR) from the skimage.metrics library:
```
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
```

Learned Perceptual Image Patch Similarity (LPIPS):
```
from lpips import LPIPS
```

For the Fréchet Inception Distance (FID), utilize the implementation available at: [Proxy-Set FID Code](https://github.com/sxzrt/Proxy-Set/tree/main/domain_gap).

#### Step3: Train SemSim and evaluate privacy leakage on reconstructed images

To assess privacy leakage through semantic similarity, you can train a model, such as ResNet18, using triplet loss on the provided [data](https://drive.google.com/file/d/1M0xnG8mHa2sZHXYrHYWlkeFLZ2XtR0Jm/view?usp=sharing). The triplet loss function helps in learning a semantically meaningful embedding space. 

```
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
loss = triplet_loss(anchor, positive, negative)
```

After training, utilize SemSim to extract features from both the original and reconstructed images. Calculate the distance between these feature sets to quantify privacy leakage. 

#### Step4: Analyse results
Calculate the correlation between different metrics and human annotated data.
```
rho, pval = spearmanr(data1,human)
print("Spearman's rank correlation coefficient: {:.4f}".format(rho))
print("p-value:", pval)

p_corr, _ = pearsonr(data1,data2)
print("Pearson rank correlation coefficient: {:.4f}".format(p_corr))

t_corr, p_value = kendalltau(data1,data2)
print("Kendall's Rank Correlation coefficient {:.4f}:".format(t_corr))
print("p-value:", p_value)
```

# Citation 

Please cite this paper if it helps your research:
```bibtex
@inproceedings{sun2023privacy,
  title={Privacy Assessment on Reconstructed Images: Are Existing Evaluation Metrics Faithful to Human Perception?},
  author={Sun, Xiaoxiao and Gazagnadou, Nidham and Sharma, Vivek and Lyu, Lingjuan and Li, Hongdong and Zheng, Liang},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

# Acknowledgement 
We express gratitude to the great work [ATSPrivacy](https://github.com/gaow0007/ATSPrivacy), [Inverting Gradients](https://github.com/JonasGeiping/invertinggradients) and [DLG](https://github.com/mit-han-lab/dlg) as we benefit a lot from both their papers and codes.

# License
This repository is released under the MIT license. 
