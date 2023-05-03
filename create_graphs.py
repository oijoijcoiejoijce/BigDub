import numpy as np
import matplotlib.pyplot as plt

x = [3, 10, 15, 30, 90]
wav2lip_psnr = [31.37, 31.37, 31.37, 31.37, 31.37]
wav2lip_ssim = [0.96, 0.96, 0.96, 0.96, 0.96]
wav2lip_fid = [36.365, 36.365, 36.365, 36.365, 36.365]

full_psnr = [31.05, 31.93, 32.55, 33.43, 33.64]
full_ssim = [0.97, 0.966, 0.972, 0.979, 0.988]
full_fid = [31.60, 31.01, 30.46, 30.3, 30.75]

finetune_psnr = [32.6, 33.19, 33.49, 33.63, 33.96]
finetune_ssim = [0.973, 0.976, 0.979, 0.979, 0.981]
finetune_fid = [30.8, 30.15, 30.00, 30.016, 30.228]

x = np.array(x)
wav2lip_psnr = np.array(wav2lip_psnr)
wav2lip_ssim = np.array(wav2lip_ssim)
wav2lip_fid = np.array(wav2lip_fid)

full_psnr = np.array(full_psnr)
full_ssim = np.array(full_ssim)
full_fid = np.array(full_fid)

finetune_psnr = np.array(finetune_psnr)
finetune_ssim = np.array(finetune_ssim)
finetune_fid = np.array(finetune_fid)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

plt.plot(x, wav2lip_psnr, label='Wav2Lip')
plt.plot(x, full_psnr, label='Full Training')
plt.plot(x, finetune_psnr, label='Finetuning')
plt.xlabel('Duration (seconds)')
plt.ylabel('PSNR')
plt.title('PSNR vs Duration (Higher is better)')

plt.legend()
plt.show()

plt.plot(x, wav2lip_ssim, label='Wav2Lip')
plt.plot(x, full_ssim, label='Full Training')
plt.plot(x, finetune_ssim, label='Finetuning')
plt.xlabel('Duration (seconds)')
plt.ylabel('SSIM')
plt.title('SSIM vs Duration (Higher is better)')

plt.legend()
plt.show()

plt.plot(x, wav2lip_fid, label='Wav2Lip')
plt.plot(x, full_fid, label='Full Training')
plt.plot(x, finetune_fid, label='Finetuning')
plt.xlabel('Duration (seconds)')
plt.ylabel('FID')
plt.title('FID vs Duration (Lower is better)')
plt.legend()
plt.show()
