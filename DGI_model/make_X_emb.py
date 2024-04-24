from encoder import *
from train import *


def make_X_emb(encoder, dataset, X_emb_path):
    t_df = None
    names = []
    name_ind = 0 # в конец эмбеддинга доьавляется номер его модели
    for data in dataset:
        encoder.eval()
        x_gpu = data.X.to(device)
        a_gpu = data.A.to(device)
        emb = encoder(x_gpu, a_gpu).detach().cpu()
        name = '_'.join(data.A_name.split('_')[:-3])
        if name in names:
            emb = torch.cat((emb, torch.tensor([[name_ind]])), dim=-1)
        else:
            names.append(name)
            name_ind += 1
            emb = torch.cat((emb, torch.tensor([[name_ind]])), dim=-1)

        if t_df is not None:
            t_df = torch.cat((t_df, emb), dim=0)
        else:
            t_df = emb

        torch.cuda.empty_cache()
        del x_gpu, a_gpu
        gc.collect()
        
    print('Finished!')
    torch.save(t_df, X_emb_path)
