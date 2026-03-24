import pandas as pd
import argparse
import os
from satimg_dataset_processor.satimg_dataset_processor import PredDatasetProcessor
from support.path_config import get_raw_data_root, get_dataset_root

RAW_DATA_DIR = str(get_raw_data_root())
DATASET_DIR = str(get_dataset_root())


def has_prediction_inputs(location_id: str) -> bool:
    location_root = os.path.join(RAW_DATA_DIR, location_id)
    viirs_day_dir = os.path.join(location_root, "VIIRS_Day")
    firepred_dir = os.path.join(location_root, "FirePred")
    if not all(os.path.isdir(path) for path in (viirs_day_dir, firepred_dir)):
        return False

    viirs_day_files = [name for name in os.listdir(viirs_day_dir) if name.endswith(".tif")]
    firepred_files = [name for name in os.listdir(firepred_dir) if name.endswith(".tif")]
    return bool(viirs_day_files) and bool(firepred_files)

dfs = []
for year in ['2017', '2018', '2019', '2020']:
    filename = 'roi/us_fire_' + year + '_out_new.csv'
    df = pd.read_csv(filename)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
dfs_test = []
for year in ['2021']:
    filename = 'roi/us_fire_' + year + '_out_new.csv'
    df_test = pd.read_csv(filename)
    dfs_test.append(df_test)
df_test = pd.concat(dfs_test, ignore_index=True)
val_ids = ['20568194', '20701026','20562846','20700973','24462610', '24462788', '24462753', '24103571', '21998313', '21751303', '22141596', '21999381', '23301962', '22712904', '22713339']

df = df.sort_values(by=['Id'])
df['Id'] = df['Id'].astype(str)
train_df = df[~df.Id.isin(val_ids)]
val_df = df[df.Id.isin(val_ids)]

train_ids = train_df['Id'].values.astype(str)
val_ids = val_df['Id'].values.astype(str)

df_test = df_test.sort_values(by=['Id'])
test_ids = df_test['Id'].values.astype(str)
test_label_sel = df_test['label_sel'].values.astype(int)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-mode', type=str, help='Train/Val/Test')
    parser.add_argument('-ts', type=int, help='Length of TS')
    parser.add_argument('-it', type=int, help='Interval')
    parser.add_argument('-limit', type=int, default=None, help='Optional limit on number of fires to process')
    args = parser.parse_args()
    ts_length = args.ts
    interval = args.it
    modes = args.mode
    if modes == 'train':
        locations = train_ids
    elif modes == 'val':
        locations = val_ids
    else:
        locations = test_ids
    locations = [location for location in locations if has_prediction_inputs(location)]
    if args.limit is not None:
        locations = locations[:args.limit]
    usecase='pred'
    satimg_processor = PredDatasetProcessor()
    if modes in ['train', 'val']:
        satimg_processor.pred_dataset_generator_seqtoseq(mode=modes, locations=locations, visualize=False, data_path=RAW_DATA_DIR,
                                                file_name=usecase+'_'+modes+'_img_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy',
                                                label_name=usecase+'_'+modes+'_label_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy',
                                                save_path = os.path.join(DATASET_DIR, 'dataset_'+modes), ts_length=ts_length, 
                                                interval=interval, image_size=(256, 256))
    else:
        for i, id in enumerate(locations):
            print(id)
            satimg_processor.pred_dataset_generator_seqtoseq(mode = 'test', locations=[id], visualize=False, data_path=RAW_DATA_DIR,file_name=usecase+'_'+id+'_img_seqtoseql_'+str(ts_length)+'i_'+str(interval)+'.npy', label_name=usecase+'_'+id+'_label_seqtoseql_'+str(ts_length)+'i_'+str(interval)+'.npy',
                                                           save_path=os.path.join(DATASET_DIR, 'dataset_test'), ts_length=ts_length, interval=interval, rs_idx=0.3, cs_idx=0.3, image_size=(256, 256), label_sel=test_label_sel[i])
