import os
from pathlib import Path
import requests

class InstrumentClient:
    def __init__(self, host, port=5000, token=None, timeout=300):
        if token is None:
            token = os.environ.get("NANOINDENT_TOKEN", "")
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"X-API-Token": token})

    def ping(self):
        r = self.session.get(f"{self.base_url}/ping", timeout=30)
        r.raise_for_status()
        return r.json()

    def call_auto(self, method, *args, **kwargs):
        payload = {
            "method": method,
            "args": list(args),
            "kwargs": kwargs,
        }
        r = self.session.post(
            f"{self.base_url}/call_auto",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
    
    # def automate_alignment_area1(self):
    #     r = self.session.post(
    #         f"{self.base_url}/automate_alignment_area1",
    #         timeout=self.timeout,
    #     )
    #     r.raise_for_status()
    #     return r.json()

    def automate_alignment_area1(
        self,
        macro_image_path,
        macro_scale_x=0.1556,
        macro_scale_y=0.1556,
        area_name="Area1",
    ):
        macro_image_path = Path(macro_image_path)

        with open(macro_image_path, "rb") as f:
            files = {
                "macro_image": (macro_image_path.name, f, "image/png")
            }
            data = {
                "macro_scale_x": str(macro_scale_x),
                "macro_scale_y": str(macro_scale_y),
                "area_name": area_name,
            }

            r = self.session.post(
                f"{self.base_url}/automate_alignment_area1",
                files=files,
                data=data,
                timeout=self.timeout,
            )

        r.raise_for_status()
        return r.json()

    def download_last_export(self, local_dir="."):
        r_info = self.session.get(f"{self.base_url}/last_export", timeout=30)
        r_info.raise_for_status()
        info = r_info.json()

        filename = info["name"]
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / filename

        r = self.session.get(
            f"{self.base_url}/download_last_export",
            stream=True,
            timeout=self.timeout,
        )
        r.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return {"status": "ok", "local_path": str(local_path), "filename": filename}

    def download_file(self, directory, filename, local_dir="."):
        payload = {"directory": directory, "filename": filename}
        r = self.session.post(
            f"{self.base_url}/download_file",
            json=payload,
            stream=True,
            timeout=self.timeout,
        )
        r.raise_for_status()

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / filename

        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return {"status": "ok", "local_path": str(local_path), "filename": filename}
    

    def restore_alignment(
        self,
        area_name,
        new_origin_macro,
        new_origin_micro,
        macro_scale_x,
        macro_scale_y,
        macro_image_path=None,
    ):
        payload = {
            "area_name": area_name,
            "new_origin_macro": list(new_origin_macro),
            "new_origin_micro": list(new_origin_micro),
            "macro_scale_x": float(macro_scale_x),
            "macro_scale_y": float(macro_scale_y),
            "macro_image_path": macro_image_path,
        }

        r = self.session.post(
            f"{self.base_url}/restore_alignment",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

        
    def move_to_aligned_point(self, area_name, x, y, plane_params=None):
        payload = {
            "area_name": area_name,
            "x": float(x),
            "y": float(y),
            "plane_params": plane_params,
        }

        r = self.session.post(
            f"{self.base_url}/move_to_aligned_point",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    
    def move_to_aligned_point_abs(self, area_name, x, y, plane_params=None):
        payload = {
            "area_name": area_name,
            "x": float(x),
            "y": float(y),
            "plane_params": plane_params,
        }

        r = self.session.post(
            f"{self.base_url}/move_to_aligned_point_abs",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
    

    def select_origin_from_rotated_image(
        self,
        image_path,
        rotation_angle,
        save_path=None,
        cmap="gray",
    ):
        import numpy as np
        # import matplotlib.pyplot as plt
        from PIL import Image
        from scipy.ndimage import rotate
        import json
        from pathlib import Path

        import matplotlib
        matplotlib.use("QtAgg")

        import matplotlib.pyplot as plt

        image_path = Path(image_path)

        img = np.array(Image.open(image_path))

        # rotate image using saved alignment angle
        rotated = rotate(
            img,
            angle=float(rotation_angle),
            reshape=True,
            order=1,
            mode="constant",
            cval=0
        )

        plt.figure(figsize=(12, 10))
        if rotated.ndim == 2:
            plt.imshow(rotated, cmap=cmap)
        else:
            plt.imshow(rotated)

        plt.title("Click the NEW ORIGIN on the rotated/aligned image")
        plt.axis("on")

        pts = plt.ginput(1, timeout=0)
        plt.close()

        if len(pts) == 0:
            raise RuntimeError("No point selected.")

        x, y = pts[0]

        result = {
            "image_path": str(image_path),
            "rotation_angle": float(rotation_angle),
            "new_origin_macro": [float(x), float(y)],
        }

        if save_path is not None:
            save_path = Path(save_path)
            with open(save_path, "w") as f:
                json.dump(result, f, indent=4)
            print(f"Saved origin info to: {save_path}")

        print("Selected origin on rotated image:")
        print(result)

        return result

    def autofocus_video(
        self,
        coarse_hold=0.6,
        fine_hold=0.1,
        coarse_steps=8,
        fine_steps=8,
        settle_time=0.5,
        drop_fraction=0.03,
        panel_rect=(232, 134, 1419, 1037),
        z_min=9.8,
        z_max=10.2,
    ):
        return self.call_auto(
            "autofocus_video",
            coarse_hold=coarse_hold,
            fine_hold=fine_hold,
            coarse_steps=coarse_steps,
            fine_steps=fine_steps,
            settle_time=settle_time,
            drop_fraction=drop_fraction,
            panel_rect=panel_rect,
            z_min=z_min,
            z_max=z_max,
        )