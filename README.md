# 🎨 Neural Style Transfer with AdaIN

> **Transform any content image into the artistic style of another image using Adaptive Instance Normalization (AdaIN).**

---

## 📌 About the Project

**Neural Style Transfer (NST)** is a deep learning technique that combines the **content of one image** with the **artistic style of another image** to generate a new stylized image.

This project implements **Arbitrary Style Transfer using Adaptive Instance Normalization (AdaIN)**.

For example:

> 🧑 Content Image + 🎨 Artistic Style Image → 🖼️ Stylized Output

Unlike traditional Neural Style Transfer methods that optimize an image separately for every style transfer, **AdaIN enables fast arbitrary style transfer using a pre-trained encoder and a trained decoder**.

The project also includes a **Flask-based web application** where users can upload a content image and a style image and generate a stylized result.

---

# ✨ Features

* 🎨 **Arbitrary Style Transfer**

  * Apply different artistic styles to different content images.

* ⚡ **Fast Style Transfer**

  * Uses AdaIN instead of optimizing the output image from scratch for every request.

* 🧠 **Deep Feature Extraction**

  * Uses a pre-trained VGG19-based encoder.

* 🔄 **Adaptive Instance Normalization**

  * Transfers the statistical characteristics of the style feature representation to the content representation.

* 🖼️ **Content Preservation**

  * Maintains the semantic structure of the original content image.

* 🎭 **Style Transformation**

  * Transfers colors, textures, and artistic patterns from the style image.

* 🌐 **Flask Web Application**

  * Upload content and style images through a browser-based interface.

* 💻 **CPU-Compatible Inference**

  * The application can run on CPU when GPU acceleration is unavailable.

---

# 🧠 How AdaIN Works

The main idea behind AdaIN is to align the feature statistics of the **content image** with those of the **style image**.

### Basic Pipeline

```text
                 Content Image
                       │
                       ▼
                ┌─────────────┐
                │ VGG19       │
                │ Encoder     │
                └──────┬──────┘
                       │
                       │ Content Features
                       ▼
                  ┌─────────┐
                  │  AdaIN  │◄──────── Style Features
                  └────┬────┘
                       │
                       │ Stylized Features
                       ▼
                ┌─────────────┐
                │   Decoder   │
                └──────┬──────┘
                       │
                       ▼
                Stylized Image
```

### Step-by-Step

1. The **content image** is passed through the VGG19 encoder.
2. The **style image** is also passed through the encoder.
3. AdaIN calculates the channel-wise mean and standard deviation of both feature representations.
4. The content features are normalized and adjusted using the statistics of the style features.
5. The resulting stylized feature representation is passed through the trained decoder.
6. The decoder reconstructs the final stylized image.

---

# 🧮 Adaptive Instance Normalization

AdaIN can be represented as:

```text
AdaIN(x, y) =
σ(y) * ((x - μ(x)) / σ(x)) + μ(y)
```

Where:

* `x` = content feature
* `y` = style feature
* `μ(x)` = channel-wise mean of content features
* `σ(x)` = channel-wise standard deviation of content features
* `μ(y)` = channel-wise mean of style features
* `σ(y)` = channel-wise standard deviation of style features

This allows the style statistics to be directly transferred to the content features.

---

# 🧠 Model Architecture

The project uses a **VGG19-based encoder** together with a trained decoder.

### Encoder

The encoder extracts high-level visual features from the input images.

```text
Input Image
     ↓
VGG19
     ↓
Convolutional Feature Layers
     ↓
Deep Feature Representation
```

The VGG19 encoder is based on a model pre-trained on **ImageNet**.

### AdaIN

```text
Content Features
       +
Style Features
       ↓
Adaptive Instance Normalization
       ↓
Stylized Feature Representation
```

### Decoder

The decoder converts the stylized feature representation back into an RGB image.

```text
Stylized Features
       ↓
Trained Decoder
       ↓
Generated Image
```

---

# 🎯 Training

The decoder is trained to reconstruct images from their normalized feature representations.

The training process uses:

### Content Loss

Content loss encourages the generated image to preserve the content structure.

```text
Content Loss =
distance between generated and target content features
```

### Style Loss

Style loss encourages the generated image to match the style statistics.

It compares the channel-wise **mean and standard deviation** of the generated and style features.

```text
Style Loss =
difference between generated and style feature statistics
```

### Total Loss

The overall objective combines content and style losses:

```text
Total Loss =
Content Weight × Content Loss
+
Style Weight × Style Loss
```

The trained decoder is then used during inference to generate stylized images.

---

# 📊 Training Configuration

The project supports configurable training parameters through command-line arguments.

Example:

```bash
python train.py \
    --experiment final_exp \
    --epochs 160 \
    --final_size 256 \
    --style_weight 5
```

For a higher-resolution experiment:

```bash
python train.py \
    --experiment final_exp \
    --epochs 200 \
    --final_size 512 \
    --style_weight 10
```

### Important Parameters

| Parameter          | Description                                 |
| ------------------ | ------------------------------------------- |
| `--experiment`     | Name of the training experiment             |
| `--epochs`         | Number of training epochs                   |
| `--final_size`     | Final image size used during training       |
| `--batch_size`     | Number of images processed per batch        |
| `--style_weight`   | Weight assigned to style loss               |
| `--content_weight` | Weight assigned to content loss             |
| `--save_interval`  | Frequency of checkpoint saving              |
| `--resume`         | Resume training from an existing checkpoint |

---

# 📁 Project Structure

```text
Neural-Style-Transfer-AdaIN/
│
├── app.py
├── train.py
├── requirements.txt
│── vgg_normalised.pth
│
├── utils/
│   ├── models.py
│   └── utils.py
│
├── content_data/
│   └── ...
│
├── style_data/
│   └── ...
│
├── experiment/
│   └── ...
│
├── static/
│   └── uploads/
│
└── templates/
    └── ...
```

### Main Files

| File                 | Purpose                                 |
| -------------------- | --------------------------------------- |
| `app.py`             | Flask web application                   |
| `train.py`           | AdaIN decoder training                  |          |
| `models.py`          | Encoder, decoder and model architecture |
| `utils.py`           | Image processing and AdaIN utilities    |
| `vgg_normalised.pth` | Pre-trained VGG19 encoder weights       |
| `content_data/`      | Content training images                 |
| `style_data/`        | Style training images                   |
| `experiment/`        | Training checkpoints                    |
| `static/uploads/`    | Uploaded/generated images               |
| `templates/`         | Flask HTML templates                    |

---

# 🛠️ Tech Stack

| Category                | Technology   |
| ----------------------- | ------------ |
| Programming Language    | **Python**   |
| Deep Learning Framework | **PyTorch**  |
| Style Transfer Method   | **AdaIN**    |
| Feature Encoder         | **VGG19**    |
| Pre-trained Weights     | **ImageNet** |
| Web Framework           | **Flask**    |
| Image Processing        | **Pillow**   |
| Numerical Computing     | **NumPy**    |
| Training Progress       | **tqdm**     |
| Production Server       | **Gunicorn** |
| Deployment              | **Render**   |

---

# 🖼️ Using the Application

### Step 1 — Upload Content Image

Upload the image whose **content/structure** you want to preserve.

Examples:

* Portrait
* Landscape
* Building
* Animal
* Photograph

### Step 2 — Upload Style Image

Upload an artistic image whose **style** you want to transfer.

Examples:

* Paintings
* Abstract art
* Watercolor
* Sketches
* Digital artwork

### Step 3 — Generate

The application processes both images using the AdaIN pipeline.

```text
Content Image
      +
Style Image
      ↓
VGG19 Encoder
      ↓
AdaIN
      ↓
Trained Decoder
      ↓
Stylized Output
```

### Step 4 — View the Result

The generated image is displayed through the Flask web application.

---

# 🚀 Deployment

The Flask application can be deployed to cloud platforms such as **Render**.

For production deployment, Gunicorn can be used:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

The application automatically uses CPU when GPU acceleration is not available.

> ⚠️ Neural Style Transfer models can require significant memory and processing time. Free cloud instances may take longer to process images and may have resource limitations.

---

# 📸 Example

The expected workflow is:

```text
┌──────────────────┐
│  Content Image   │
│                  │
│       🏙️         │
└────────┬─────────┘
         │
         │
         ▼
      🧠 AdaIN
         ▲
         │
         │
┌────────┴─────────┐
│   Style Image    │
│                  │
│       🎨         │
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ Stylized Output  │
│                  │
│   🏙️ + 🎨        │
└──────────────────┘
```

Add your actual screenshots or generated results here:

```markdown
![Content Image](examples/content.jpg)

![Style Image](examples/style.jpg)

![Stylized Output](examples/output.jpg)
```

---

# 📚 What I Learned

Through this project, I explored:

* 🧠 How **CNNs** extract visual features from images
* 🏗️ How **VGG19** can be used as a feature encoder
* 🎨 How neural style transfer works
* 📐 How **Adaptive Instance Normalization (AdaIN)** transfers style statistics
* 📉 The role of content loss and style loss
* 🔄 How encoder-decoder architectures are used for image generation
* 🔥 How to train deep learning models using **PyTorch**
* 🖼️ Image preprocessing and reconstruction
* 🌐 How to integrate a deep learning model into a **Flask web application**
* ☁️ How to deploy an AI application to the cloud

---

# 🔬 Research Reference

This implementation is based on the AdaIN approach introduced in:

> **Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization**

**Huang, X., & Belongie, S. (2017)**

The original research introduced Adaptive Instance Normalization as a method for performing real-time arbitrary style transfer.

---

# 🙏 Acknowledgements

* **Xun Huang & Serge Belongie** — for the AdaIN research paper
* **VGG19 / ImageNet** — for the pre-trained feature encoder
* **PyTorch** — deep learning framework
* **Flask** — web application framework
* Open-source contributors whose libraries made this project possible

---

# 🚀 Future Improvements

Possible improvements for future versions include:

* ⚡ Faster inference
* 🎨 Better style-content blending controls
* 🖼️ Higher-resolution output generation
* 🎚️ Adjustable style strength
* 📱 Improved responsive UI
* 🖥️ GPU acceleration for faster inference
* 📊 More extensive training datasets
* 🎭 Support for multiple style images
* 🔄 Real-time style transfer
* 📥 Downloadable high-resolution results

---

# 📄 License

This project is open source and available under the **MIT License**.

See the [`LICENSE`](LICENSE) file for details.

---

# 👨‍💻 Author

**Rohit Singh Rawat**

MCA Data Science | AI/ML/DS Enthusiast

This project was developed as part of my exploration of **Deep Learning, Computer Vision, and Generative AI**.

---

<p align="center">
  <b>🎨 Neural Style Transfer with AdaIN</b>
  <br>
  Transform images with the power of Deep Learning.
  <br><br>
  Made with ❤️ using Python & PyTorch
</p>
