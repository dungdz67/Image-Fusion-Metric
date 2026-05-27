function SSIM = SSIM_metrics(img1, img2, img_f)
    SSIM1 = ssim(img_f,img1);
    SSIM2 = ssim(img_f,img2);
    SSIM = 0.5 * SSIM1 + 0.5 * SSIM2;