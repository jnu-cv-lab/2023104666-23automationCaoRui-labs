import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p):
    if not os.path.exists(p): os.makedirs(p)


def mse(a,b): return np.mean((a-b)**2)

def psnr(a,b):
    m=mse(a,b)
    return 100 if m==0 else 10*np.log10((255**2)/m)


def fft_spectrum(img):
    f=np.fft.fft2(img)
    f=np.fft.fftshift(f)
    return np.log(1+np.abs(f))


def dct2(img): return cv2.dct(np.float32(img))


def energy_ratio(d, r=0.25):
    h,w=d.shape
    hl,wl=int(h*r),int(w*r)
    return np.sum(d[:hl,:wl]**2)/np.sum(d**2)


def save(p, img): plt.imsave(p, img, cmap='gray')


def main():
    base=os.path.dirname(__file__)
    out=os.path.join(base,'results')
    ensure_dir(out)

    img=cv2.imread(os.path.join(base,'input.jpg'),0)
    if img is None:
        print('请放 input.jpg'); return

    # 1️ 原图（1张）
    save(os.path.join(out,'1_original.png'), img)

    # 2️ 下采样（2张：1/2, 1/4）
    blur=cv2.GaussianBlur(img,(5,5),1)
    down_2=cv2.resize(blur,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_NEAREST)
    down_4=cv2.resize(blur,None,fx=0.25,fy=0.25,interpolation=cv2.INTER_NEAREST)
    save(os.path.join(out,'2_down_1_2.png'), down_2)
    save(os.path.join(out,'3_down_1_4.png'), down_4)

    # 3️ 恢复（4张：两种方法 × 两个尺度）
    def restore(down, name_prefix):
        r_lin=cv2.resize(down,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LINEAR)
        r_cub=cv2.resize(down,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_CUBIC)
        save(os.path.join(out,f'{name_prefix}_bilinear.png'), r_lin)
        save(os.path.join(out,f'{name_prefix}_bicubic.png'), r_cub)
        return r_lin, r_cub

    r2_lin, r2_cub = restore(down_2, '4_restore_1_2')
    r4_lin, r4_cub = restore(down_4, '5_restore_1_4')

    # 4️ 空间域指标（打印）
    print('--- Spatial Metrics ---')
    for name, r in [('1/2-bilinear',r2_lin),('1/2-bicubic',r2_cub),
                    ('1/4-bilinear',r4_lin),('1/4-bicubic',r4_cub)]:
        print(name, 'MSE=%.2f PSNR=%.2f'%(mse(img,r), psnr(img,r)))

    # 5️ FFT（3张：原图、1/2下采样、1/2双线性恢复）
    save(os.path.join(out,'6_fft_original.png'), fft_spectrum(img))
    save(os.path.join(out,'7_fft_down_1_2.png'), fft_spectrum(down_2))
    save(os.path.join(out,'8_fft_restore_bilinear.png'), fft_spectrum(r2_lin))

    # 6️ DCT（3张：原图、双线性、双三次）
    d_o=dct2(img)
    d_lin=dct2(r2_lin)
    d_cub=dct2(r2_cub)

    save(os.path.join(out,'9_dct_original.png'), np.log(1+np.abs(d_o)))
    save(os.path.join(out,'10_dct_bilinear.png'), np.log(1+np.abs(d_lin)))
    save(os.path.join(out,'11_dct_bicubic.png'), np.log(1+np.abs(d_cub)))

    print('\n--- DCT Energy Ratio ---')
    print('original:', energy_ratio(d_o))
    print('bilinear:', energy_ratio(d_lin))
    print('bicubic:', energy_ratio(d_cub))

    # 7️ 一张总对比图（第12张）
    fig,ax=plt.subplots(2,3,figsize=(10,6))
    ax[0,0].imshow(img,cmap='gray'); ax[0,0].set_title('Original')
    ax[0,1].imshow(down_2,cmap='gray'); ax[0,1].set_title('Down 1/2')
    ax[0,2].imshow(r2_lin,cmap='gray'); ax[0,2].set_title('Bilinear')
    ax[1,0].imshow(down_4,cmap='gray'); ax[1,0].set_title('Down 1/4')
    ax[1,1].imshow(r4_lin,cmap='gray'); ax[1,1].set_title('Bilinear 1/4')
    ax[1,2].imshow(r2_cub,cmap='gray'); ax[1,2].set_title('Bicubic')
    for a in ax.ravel(): a.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out,'12_summary.png'))
    plt.close()

    print('\n 已生成12张图片（results文件夹）')

if __name__=='__main__':
    main()
