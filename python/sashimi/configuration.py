import json
import os


class Configuration(object):
    def __init__(self):
        self.port = "COM3"
        self.home_offset = [10000, 50000, 2000]
        self.stack_height = 1000
        self.stack_step = 60
        self.exposure_time = 2000
        self.scans = [{'FL': [10000, 50000, 2000],
                       'BR':[11000, 51000, 2000],
                       'BL_Z':2000,
                       'Z_corrections':[0, 0]}]
        self.top_down = False

    def update_z_correction_terms(self, index, blz=None):
        # supposes the scan surface is flat and non-vertical
        fl, br = self.scans[index]['FL'], self.scans[index]['BR']
        x, y, z = 0, 1, 2
        
        if blz is None:
            blz = (fl[z] + br[z])//2
        
        assert (br[x] != fl[x])
        assert (br[y] != fl[y])
        
        dz_dx = (blz - fl[z]) / (br[x] - fl[x])
        dz_dy = (br[z] - blz) / (br[y] - fl[y])

        self.scans[index]['BL_Z'] = blz
        self.scans[index]['Z_corrections'] = [dz_dx, dz_dy]
        self.save()

    def save(self, save_name="config"):
        config_file = os.path.join(os.path.expanduser("~"), ".sashimi", save_name + ".json")
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, "w") as f:
            j = json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
            f.write(j)

    @staticmethod
    def load(save_name="config"):
        print("Load config")
        config_file = os.path.join(os.path.expanduser("~"), ".sashimi", save_name + ".json")
        if os.path.exists(config_file):
            print("Config file found")
            try:
                with open(config_file, "r") as f:
                    j = json.load(f)
                    print(j)
                    config = Configuration()
                    print(config.__dict__)
                    for key, val in j.items():
                        print(f"{key}: {val}")
                        if key in config.__dict__:
                            config.__dict__[key] = val
            except RuntimeError:
                config = Configuration()
        else:
            config = Configuration()
        return config
