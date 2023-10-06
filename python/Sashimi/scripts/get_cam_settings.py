from pypylon import pylon

if __name__ == '__main__':
    nodeFile = "nodeFile.pfs"
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    pylon.FeaturePersistence_Save(nodeFile, camera.GetNodeMap())
    camera.Close()
