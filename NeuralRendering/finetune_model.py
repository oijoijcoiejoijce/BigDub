from NeuralRendering.model import NeuralRenderer
from Datasets import DubbingDataset, DataTypes
import torch
import os
import cv2

torch.backends.cudnn.benchmark = True

def test_model(dataset, model, video_name, save_path):

    with torch.no_grad():
        gen, length = dataset.get_video_generator(video_name)
        video = model.create_video_from_generator(gen, length)

        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video.shape[3], video.shape[2]))
        for frame in video:
            frame = frame.transpose(1, 2, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)

def texture_transfer_learning(dataset, model, ID, save_root):

    model.fit_to_new_ID(dataset, n_epoch=1, ID_name=ID)

def finetune_model(dataset, model, ID, save_root):

    model.finetune(dataset, n_epoch=1, ID_name=ID)

def main(config, model_checkpoint, ID, videos, val_videos, save_root):

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    data_root = config.get('Paths', 'data')
    dataset = DubbingDataset(data_root,
                             data_types=[DataTypes.MEL, DataTypes.Params, DataTypes.Frames, DataTypes.ID], split='all',
                             T=5, fix_ID=ID, restrict_videos=videos)

    validation_dataset = DubbingDataset(data_root,
                            data_types=[DataTypes.MEL, DataTypes.Params, DataTypes.Frames, DataTypes.ID], split='all',
                            T=5, fix_ID=ID, restrict_videos=val_videos)

    model = NeuralRenderer.load_from_checkpoint(model_checkpoint, config=config,
                                                IDs=dataset.ids, strict=False)
    #model.eval()
    model.cuda()

    for epoch in range(10):

        # Finetune
        texture_transfer_learning(dataset, model, ID, save_root)

        # Validate
        #if not os.path.exists(os.path.join(save_root, f'epoch_{epoch}')):
        #    os.makedirs(os.path.join(save_root, f'epoch_{epoch}'))
        #for video in val_videos:
        #    save_path = os.path.join(save_root, f'epoch_{epoch}', video + '.mp4')
        #    test_model(validation_dataset, model, video, save_path)

    for epoch in range(10):

        # Finetune
        finetune_model(dataset, model, ID, save_root)

    # Validate
    if not os.path.exists(os.path.join(save_root, f'videos')):
        os.makedirs(os.path.join(save_root, f'videos'))
    for video in val_videos:
        save_path = os.path.join(save_root, f'videos', video + '.mp4')
        test_model(validation_dataset, model, video, save_path)

    for video in val_videos:
        test_model(validation_dataset, model, video, save_root)


if __name__ == '__main__':
    import configparser
    config = configparser.ConfigParser()

    config_path = 'configs/Laptop.ini'
    config.read(config_path)

    # checkpoint = "C:/Users/jacks/Documents/Data/DubbingForExtras/checkpoints/render/epoch=99-step=186300.ckpt"
    checkpoint = "C:/Users/jacks/Documents/Data/DubbingForExtras/checkpoints/render/epoch=199-step=496800.ckpt"

    for ID in ['M009', 'M030', 'W011']:

        for n_vid in [1, 3, 5, 10, 30]:

            #ID = 'M009'
            video_idxs = list(range(n_vid)) #[0] #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            videos = [f'{ID}_{i}' for i in video_idxs]
            val_videos_idxs = [35, 36, 37, 38, 39]  #[f'{ID}_{i}' for i in range(40) if i not in video_idxs]
            val_videos = [f'{ID}_{i}' for i in val_videos_idxs]

            save_root = f"C:/Users/jacks/Documents/Data/DubbingForExtras/test/finetune_{ID}_{len(videos)}_videos"

            main(config, checkpoint, ID, videos, val_videos, save_root)
