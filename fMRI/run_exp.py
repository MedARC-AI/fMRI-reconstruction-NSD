import papermill as pm
from sklearn.model_selection import ParameterGrid

if __name__ == '__main__':
    grid = dict(
        model_name=['clip_image_vit', 'clip_text_vit', 'clip_image_resnet'],
        clip_vit_version=['ViT-L/14', 'ViT-B/32'],
        use_img_aug=[True, False],
    )
    grid = ParameterGrid(grid)

    for params in list(grid):
        print(params)
        tag = '|'.join([f'{k}={v}' for k, v in params.items()])
        print(tag)

        # res = pm.execute_notebook(
        #     'Voxel_to_CLIPvoxel.ipynb',
        #     f'Voxel_to_CLIPvoxel_{tag}.ipynb',
        #     parameters=params
        # )