import os
import cv2
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


class UNet(nn.Module):
    """U-Net model for brain tumor segmentation from MRI images."""

    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Contracting path (encoder)
        self.conv1 = nn.Conv2d(self.n_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Expansive path (decoder)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(64, self.n_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass of U-Net."""
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(self.pool(x1)))
        x3 = F.relu(self.conv3(self.pool(x2)))
        x4 = F.relu(self.conv4(self.pool(x3)))
        x5 = F.relu(self.conv5(self.pool(x4)))

        x6 = F.relu(self.upconv1(x5))
        x6 = torch.cat([x4, x6], dim=1)
        x6 = F.relu(self.conv6(x6))
        x7 = F.relu(self.upconv2(x6))
        x7 = torch.cat([x3, x7], dim=1)
        x7 = F.relu(self.conv7(x7))
        x8 = F.relu(self.upconv3(x7))
        x8 = torch.cat([x2, x8], dim=1)
        x8 = F.relu(self.conv8(x8))
        x9 = F.relu(self.upconv4(x8))
        x9 = torch.cat([x1, x9], dim=1)
        x9 = F.relu(self.conv9(x9))
        x10 = self.conv10(x9)

        return x10


class BrainTumorAgent:
    """
    Handles brain tumor detection and segmentation from MRI images using a
    trained U-Net model.

    If the model checkpoint is not available locally (and no valid download id
    is provided), the agent degrades gracefully: model loading is deferred and
    ``predict`` returns ``None`` instead of crashing the whole pipeline.
    """

    # Google Drive file id for the pre-trained brain tumor segmentation model.
    # Leave as None if unavailable; the agent will then run in degraded mode.
    GDRIVE_MODEL_ID = None

    def __init__(self, model_path=None, device=None):
        default_path = os.path.join(
            os.path.dirname(__file__), "models", "brain_tumor_segmentation.pth"
        )
        self.model_path = model_path or default_path
        self.device = device if device else DEVICE
        self.model = self._load_model()

    def _load_model(self):
        """Load the trained U-Net model if the checkpoint is available."""
        try:
            # Try to download the checkpoint if a valid Google Drive id is set.
            if self.GDRIVE_MODEL_ID and not os.path.exists(self.model_path):
                from .model_download import download_model_checkpoint

                download_model_checkpoint(self.GDRIVE_MODEL_ID, self.model_path)

            if not os.path.exists(self.model_path):
                logger.warning(
                    "Brain tumor model checkpoint not found at "
                    f"'{self.model_path}'. Running in degraded mode "
                    "(no inference will be performed)."
                )
                return None

            model = UNet(n_channels=3, n_classes=1).to(self.device)
            checkpoint = torch.load(self.model_path, map_location=torch.device(self.device))
            # Support both raw state_dict and checkpoint dicts with 'state_dict'.
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            logger.info(f"Brain tumor model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading brain tumor model: {e}")
            # Degrade gracefully instead of raising, so the app can still run.
            return None

    def _overlay_mask(self, img, mask, output_path):
        """Overlay the segmentation mask on the original MRI image and save it."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            mask_stacked = np.stack((mask,) * 3, axis=-1)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.axis("off")
            ax.imshow(img)
            ax.imshow(mask_stacked, alpha=0.4, cmap="Reds")
            plt.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Overlayed brain tumor segmentation mask saved at '{output_path}'")
            return True
        except Exception as e:
            logger.error(f"Error generating brain tumor overlay: {e}")
            return False

    def predict(self, image_path, output_path):
        """
        Segment tumor region in a brain MRI image and save the overlaid
        visualization.

        Returns:
            True  -> segmentation succeeded and output was written
            False -> inference failed (unclear / non-medical image or IO error)
            None  -> model unavailable (degraded mode)
        """
        if self.model is None:
            logger.warning("Brain tumor model is unavailable; skipping inference.")
            return None

        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                logger.error(f"Could not read image at '{image_path}'.")
                return False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0,1]
            img_resized = cv2.resize(img, (256, 256))
            img_tensor = (
                torch.Tensor(img_resized).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            )

            with torch.no_grad():
                logits = self.model(img_tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            # Binarize the predicted probability map into a mask.
            generated_mask = (probs > 0.5).astype(np.float32)

            # Resize mask to match original image dimensions.
            generated_mask_resized = cv2.resize(
                generated_mask, (img.shape[1], img.shape[0])
            )
            return self._overlay_mask(img, generated_mask_resized, output_path)

        except Exception as e:
            logger.error(f"Error during brain tumor segmentation: {e}")
            return False


# # Example Usage
# if __name__ == "__main__":
#     agent = BrainTumorAgent(model_path="./models/brain_tumor_segmentation.pth")
#     result = agent.predict("./images/brain_mri.jpg", "./brain_tumor_plot.png")
#     logger.info(f"Brain tumor segmentation result: {result}")
