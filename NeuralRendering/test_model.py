from NeuralRendering.model import NeuralRenderer
from Datasets import DubbingDataset, DataTypes
import torch
import os
import cv2

def test_model(dataset, model, video_name, save_root):

    gen, length = dataset.get_video_generator(video_name)

    save_path = os.path.join(save_root, video_name + '.mp4')
    video = model.create_video_from_generator(gen, length)

    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video.shape[3], video.shape[2]))
    for frame in video:
        frame = frame.transpose(1, 2, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()


def main(config, model_checkpoint, videos, save_root):

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    data_root = config.get('Paths', 'data')
    dataset = DubbingDataset(data_root,
                             data_types=[DataTypes.MEL, DataTypes.Params, DataTypes.Frames, DataTypes.ID], split='all',
                             T=5)

    model = NeuralRenderer.load_from_checkpoint(model_checkpoint, config=config, IDs=dataset.ids, strict=False)
    #model.eval()
    model.cuda()

    for video in videos:
        print(video)
        test_model(dataset, model, video, save_root)
        print('Done')

if __name__ == '__main__':
    import configparser
    config = configparser.ConfigParser()

    config_path = 'configs/Laptop.ini'
    config.read(config_path)

    model = "M009_03"

    for model in ["M009_01", "M009_30", "W011_10", "W011_30"]:

        checkpoint = f"C:/Users/jacks/Documents/Data/DubbingForExtras/checkpoints/PersonSpecificModels/{model}.ckpt"
        videos = [f'{model.split("_")[0]}_{i:02d}' for i in range(35, 40)]

        save_root = f"C:/Users/jacks/Documents/Data/DubbingForExtras/test/{model}"

        main(config, checkpoint, videos, save_root)
