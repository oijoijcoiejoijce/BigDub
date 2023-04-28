import pytorch_lightning

from NeuralRendering.model import NeuralRenderer
from Datasets import DubbingDataset, DataTypes
import torch
import os
import cv2

torch.backends.cudnn.benchmark = True

def test_model(dataset, model, video_name, save_root):

    best_model_path = os.path.join(save_root, 'finetune.pt')
    model.load_finetune(best_model_path)

    save_path = os.path.join(save_root, f'videos', f'{video_name}.mp4')

    with torch.no_grad():
        gen, length = dataset.get_video_generator(video_name)
        video = model.create_video_from_generator(gen, length)

        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video.shape[3], video.shape[2]))
        for frame in video:
            frame = frame.transpose(1, 2, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()

def texture_transfer_learning(dataset, model, ID, save_root, val_loader):

    model.fit_to_new_ID(dataset, n_epoch=1, ID_name=ID, save_dir=save_root, val_loader=val_loader)

def finetune_model(dataset, model, ID, save_root, val_loader):

    model.finetune(dataset, n_epoch=1, ID_name=ID, save_dir=save_root, val_loader=val_loader)

def main(config, model_checkpoint, ID, videos, val_videos, test_videos, save_root):

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    data_root = config.get('Paths', 'data')
    dataset = DubbingDataset(data_root,
                             data_types=[DataTypes.MEL, DataTypes.Params, DataTypes.Frames, DataTypes.ID], split='all',
                             T=5, fix_ID=ID, restrict_videos=videos)

    validation_dataset = DubbingDataset(data_root,
                            data_types=[DataTypes.MEL, DataTypes.Params, DataTypes.Frames, DataTypes.ID], split='all',
                            T=5, fix_ID=ID, restrict_videos=val_videos)
    test_dataset = DubbingDataset(data_root,
                            data_types=[DataTypes.MEL, DataTypes.Params, DataTypes.Frames, DataTypes.ID], split='all',
                            T=5, fix_ID=ID, restrict_videos=test_videos)

    model = NeuralRenderer.load_from_checkpoint(model_checkpoint, config=config,
                                                IDs=dataset.ids, strict=False)
    model.trainer = pytorch_lightning.Trainer(gpus=1)
    #model.eval()
    model.cuda()

    for epoch in range(25):

        # Finetune
        texture_transfer_learning(dataset, model, ID, save_root, validation_dataset)

    #for epoch in range(40):
    #    # Finetune
    #    finetune_model(dataset, model, ID, save_root, validation_dataset)

    # Validate
    if not os.path.exists(os.path.join(save_root, f'videos')):
        os.makedirs(os.path.join(save_root, f'videos'))

    for i, video in enumerate(test_videos):

        test_model(test_dataset, model, video, save_root)


if __name__ == '__main__':
    import configparser
    config = configparser.ConfigParser()

    config_path = 'configs/Laptop.ini'
    config.read(config_path)

    # checkpoint = "C:/Users/jacks/Documents/Data/DubbingForExtras/checkpoints/render/epoch=99-step=186300.ckpt"
    checkpoint = "C:/Users/jacks/Documents/Data/DubbingForExtras/checkpoints/render/epoch=199-step=496800.ckpt"

    for ID in ['M030', 'W011']:  # 'M009'

        for n_vid in [1, 3, 5, 10, 30]:

            if ID == 'M030' and n_vid == 1:
                continue


            video_idxs = list(range(n_vid))
            videos = [f'{ID}_{i}' for i in video_idxs]

            val_videos = [30, 31, 32, 33, 34]
            val_videos = [f'{ID}_{i}' for i in val_videos]

            test_videos_idxs = [35, 36, 37, 38, 39]
            test_videos = [f'{ID}_{i}' for i in test_videos_idxs]

            save_root = f"C:/Users/jacks/Documents/Data/DubbingForExtras/test/finetune_tex_{ID}_{len(videos)}_videos"

            main(config, checkpoint, ID, videos, val_videos, test_videos, save_root)
