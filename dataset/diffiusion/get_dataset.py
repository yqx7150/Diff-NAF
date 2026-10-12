from dataset.diffiusion.dataset_pickel import BaseDataset as pickle_data_load

def dataset_pickle(config ,condition=0,phese = 'train',data_type='AAPMDR'):

    return pickle_data_load(config , type=phese,condition=condition,split_ratio=0.99)