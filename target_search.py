import numpy as np
import pandas as pd
import os,sys
import joblib
import matplotlib.pyplot as plt
from sklearn import model_selection, ensemble, preprocessing, metrics

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from tqdm import tqdm 

vers = 1.0 # atomref #eng-only mode
scri = 'rf_target' # define output name

def preprocess():

  pref = {}
  pref['run'] = 'train'  ## predict, train
  pref["train"] = "rs_2000.csv"
  pref["test"] = 'default'
  pref["model"] = 'default'
  pref['ml'] = 'rf'
  
  #pref["feat"] = ['dih1_i','dih2_i','cp1p_i','cp1t_i','cp2p_i','cp2t_i','ionCRD_i','HB_i'] #,'E_i']
  pref["feat"] = ['dih1_i','dih2_i','s_cp1p','s_cp1t','b_cp1p','b_cp1t','ionCRD_i','HB_i'] #,'E_i']
  #pref["feat"] = ['dih1','dih2','cp1p','cp1t','cp2p','cp2t','ionCRD','E_i'] 
  #pref["feat"] = ['dih1','dih2','E_f'] 
  
  pref["pred"] = 'label' 
  pref['target'] = [30,80]
  
  pref['trial'] = 100
  pref['esti'] = 10
  pref['clf'] = 1
  pref['ion_pos'] = [0,1,2,3,4,6]
  #pref['read_model'] = 0
  #pref['premodel'] = 'rfModel.joblib'
  
  pref['xlim'] = [0,200]
  pref['device'] = 'cpu'
  pref['batch'] = 100
  pref['lr'] = 0.0004
  pref['epochs'] = 100
  
  # ======= Preprocessing ================
  pref_keys = [i for i in pref.keys()]
  for ar in sys.argv[1:]:
    ar = ar.split('=')
    if (len(ar) > 1) & (ar[0] in pref_keys):
      if type(pref[ar[0]]) == list:
        pref[ar[0]] = eval(ar[1])
      else:
        pref[ar[0]] = type(pref[ar[0]])(ar[1])
    else:
      print ('keys= \n',pref,'\n')
      exit()
      
  if pref['run'] == 'predict' and pref["model"] == 'default':
    model = [ i for i in os.listdir('./') if i[-6:] == 'joblib' ]
    if len(model) == 1:
      pref["model"] = model[0]
      print ('premodel = ', pref["model"])      
    elif len(model) == 0:
      print ('No RF model found')
      exit()      
    else:
      k = [ print (f"{i}") for i in model ]
      print ('Multiple models are available; please assign the one to be used.\n')
      exit()
      
  if pref["test"] == 'default':
    pref["test"] = pref["train"].replace('.csv','') + '_test.csv'
  
  prefix = pref['train'].split('.')[0] 
  pref['pred_fn'] = f"{prefix}_{pref['target'][0]}_{pref['target'][1]}"  
  pref['clf'] = 1 if pref["pred"] == 'label' else 0
    
  print (f"test_set = {pref['test']}")
  return pref

  
  
####################################  
def dt_label(df):
  lab = []
  for e in df['E'].tolist():
    lab.append(int(e < pref['target'][1]))
  return lab
  
def format_CRD(df,pf):  
  oxy = pref['ion_pos']
  lab = [[] for i in range(len(oxy))]
  for ion in df[pf].tolist():
    ion = eval(ion)
    for xi, x in enumerate(oxy):
      if x in ion:
        lab[xi].append(1)
      else:
        lab[xi].append(0)
        
  ion_cols = []  
  for xi, x in enumerate(oxy):
    df[f"ion_{x}"] = lab[xi]
    ion_cols.append(f"ion_{x}")
  return df, ion_cols

def format_HB(df, pf, hb_col=[]):
  hb_ls = []
  detectHB = 0
  if not hb_col:
    detectHB = 1
  else:
    hb_col = [i[3:] for i in hb_col]
    hb_ls= [[] for i in range(len(hb_col))]
      
  for hb in df[pf].tolist():
    hb = eval(hb)
    if detectHB:
      for h in hb:
        if h not in hb_col:
          if hb_ls: 
            hb_ls.append([0 for i in range(len(hb_ls[0]))])
          else:
            hb_ls.append([])
          hb_col.append(h) 
     
    for hi, h in enumerate(hb_col):      
      if h in hb:
        hb_ls[hi].append(1)
      else:       
        hb_ls[hi].append(0) 
  for xi, x in enumerate(hb_col):
    df[f"hb_{x}"] = hb_ls[xi]
    hb_col[xi] = f"hb_{x}"
  return df, hb_col
  
def try_rm(val, ls):
  try:
    ls.remove(val)
  except:
    pass
  return ls
  
def prepare_dataset(data_csv, feat=[]):
  dt = pd.read_csv(data_csv)
  ext = 0
  if not feat:
    feat = pref["feat"].copy()
    ext = 1
       
  for pf in pref["feat"]:
    if 'ionCRD' in pf:
      dt, crd_col = format_CRD(dt, pf)  
      feat = try_rm(pf, feat)
      if ext:
        feat.extend(crd_col)
      
    elif 'HB' in pf:
      dt, hb_col = format_HB(dt, pf, [i for i in feat if i[:3] == 'hb_'])  
      feat = try_rm(pf, feat)   
      if ext:
        feat.extend(hb_col)
      
  if pref['pred'] == 'label' and pref["pred"] not in dt.columns: 
    dt[pref["pred"]] = dt_label(dt)  

  return dt, feat
  
def prepare_test(feature):
  test_set, f_test = prepare_dataset(pref["test"], feature)  
  test_x = test_set[feature]
  test_y = test_set[pref['pred']].tolist()
  if pref['clf']:
    test_y_lab = test_y 
  else:
    test_y_lab = [int(i < pref['target'][1]) for i in test_y]
  dt_pred = pd.DataFrame({'act':test_y, 'act_l':test_y_lab, 'E':test_set['E']})
  
  for i in [10,5,3]:
    print(f"test {i}%= {round(dt_pred['E'].quantile(i/100),2)}")    
  return test_x, dt_pred
  
####################################
def hist_fig(dt):
  left, width = 0.13, 0.8
  bottom, height = 0.10, 0.85
  spacing = 0.005  
  
  rect_hist = [left, bottom, width, height]
  plt.figure(figsize=(7, 4), dpi=600)
  ax_hist = plt.axes(rect_hist)
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)
 
  vmax = int(dt['E'].max())
  vmin = int(dt['E'].min())
  vstep = 5 
  bins = [ i*vstep for i in range(int(vmin/vstep)-1,int(vmax/vstep)+1)] 
    
  ax_hist.hist(dt['E'], bins=bins,color='grey',histtype=u'step')
  dt2 = dt[dt['pred_l'] == 1].copy()
  ax_hist.hist(dt2['E'], bins=bins,color='grey')
  y = ax_hist.get_ylim()
  plt.plot([pref['target'][0],pref['target'][0]],y)
  plt.plot([pref['target'][1],pref['target'][1]],y)
  
  if pref['xlim']:
    plt.xlim(pref['xlim'][0],pref['xlim'][1])
   
  fn = f"plot_rf_hist{pref['target'][0]}-{pref['target'][1]}.png"
  plt.savefig(fn,format='png')   
 
####################################   
def label_evaluate(dt_p, best, final=0):   
  overall_acc = round(100* metrics.accuracy_score(dt_p['act_l'], dt_p['pred_l']),1)    
   
  dt_cutoff = dt_p[dt_p['act_l'] == 1]
  num_label = len(dt_p[dt_p['pred_l'] == 1])
  dt_lowE = dt_cutoff[dt_cutoff['E'] < pref['target'][0]]
  correct_lowE = len(dt_lowE[dt_lowE['pred_l'] == 1])
  num_lowE = len(dt_lowE)
  acc_score = correct_lowE / num_lowE 

  if acc_score >= best:
    best = acc_score
    correct_cutoff = len(dt_cutoff[dt_cutoff['pred_l'] == 1])
    num_cutoff = len(dt_cutoff)
    acc_cutoff = round(100 * correct_cutoff / num_cutoff,1)
    
    percent_total_label = round(100 * num_label / len(dt_p),1)
    acc_lowe = round(100 * correct_lowE / num_lowE, 1)
    
    if final:
      dt_p.to_csv(f"pred_{pref['pred_fn']}.csv", index=False)
      #candidate = test_set[[i for i in test_y_pred]]  
      #candidate.to_csv(f"info_lab_{pref['pred_fn']}.csv", index=False)
    
    print ("\nEvaluation:")
    if not pref['clf']:
      print (f"MAE= {round(np.mean(abs(dt_p['act'] - dt_p['pred'])),2)}\n")
    print ("test_set\t\tacc_cut\tacc_LE\tpercent_Label")
    print (f"{pref['test']}\t{acc_cutoff}%\t{acc_lowe}%\t{percent_total_label}%\n")    
    print (f"LLE\tLE\tL")
    print (f"{correct_lowE}\t{num_lowE}\t{num_label}\n\n") 
            
    fw = open('rf.log','a')
    fw.writelines(f"{pref['test']},{pref['target'][0]},{pref['target'][1]},{overall_acc},{acc_cutoff},{acc_lowe},{percent_total_label},{correct_cutoff},{num_cutoff},{correct_lowE},{num_lowE},{num_label}\n")
    fw.close()     
  return best
  
####################################  
def forest_predict(forest, test_x, dt_p):
  test_y_pred = forest.predict(test_x)  
  if pref['clf']:
    test_y_lab_pred = test_y_pred
  else:
    test_y_lab_pred = [ int(i < pref['target'][1]) for i in test_y_pred]
  dt_p['pred'] = test_y_pred
  dt_p['pred_l'] = test_y_lab_pred  
  return dt_p  
  
def rf_train():
  ### train set
  train_set, feature = prepare_dataset(pref["train"])
  train_x = train_set[feature]
  train_y = train_set[pref["pred"]].tolist()  
  
  print (f"input feature= \n {pref['feat']}\n")
  print (f"expended feature= \n {feature}\n")
  
  ### test set
  test_x, dt_pred = prepare_test(feature)  
  
  ### RF training
  best = 0       
  for trial in range(pref['trial']): 
    if trial % 20 == 0:
      print (f"training trial {trial}")
       
    if pref['clf']:
      forest = ensemble.RandomForestClassifier(bootstrap=True, n_estimators=pref['esti'])
    else:
      forest =  ensemble.RandomForestRegressor(n_estimators=pref['esti'], oob_score=False)  
      
    forest.feature_names = feature
    forest_fit = forest.fit(train_x, train_y)
    
    dt_pred = forest_predict(forest, test_x, dt_pred)
    best_i = label_evaluate(dt_pred, best)
    
    if best_i > best:
      joblib.dump(forest, f"rf_{pref['pred_fn']}.joblib")
      best = best_i
      best_model = forest   
  rf_evaluate(best_model, test_x, dt_pred)            
      
def rf_evaluate(model, test_x=[], dt_pred=[]):       
  feature = model.feature_names
  if len(test_x) == 0:  
    test_x, dt_pred = prepare_test(feature)
         
  ### RF predict
  dt_pred = forest_predict(model, test_x, dt_pred)
  best = label_evaluate(dt_pred, 0, 1)
  hist_fig(dt_pred)
    
####################################
def fit_model(model, train_loader, test_loader, epochs, lr):
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(pref['device'])
    history = {'train_loss': [], 'val_loss': []}

    print ('\nStart training:')
    for epoch in range(epochs):
        model.train() 
        tloss = 0.0
        for x1d_batch, x2d_batch, y_batch in train_loader:
            x1d_batch = x1d_batch.to(pref['device'])
            x2d_batch = x2d_batch.to(pref['device']) 
            y_batch = y_batch.to(pref['device'])

            optimizer.zero_grad() 
            predictions = model(x1d_batch, x2d_batch)
            loss = criterion(predictions.squeeze(), y_batch)  
            loss.backward() 
            optimizer.step()         
            tloss += loss.item()

        tloss /= len(train_loader)
        history['train_loss'].append(tloss)
        
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for x1d_batch, x2d_batch, y_batch in test_loader:
                x1d_batch = x1d_batch.to(pref['device'])
                x2d_batch = x2d_batch.to(pref['device']) 
                y_batch = y_batch.to(pref['device'])

                predictions = model(x1d_batch, x2d_batch)
                loss = criterion(predictions.squeeze(), y_batch)
                vloss += loss.item()

        vloss /= len(test_loader)
        history['val_loss'].append(vloss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: train= {tloss:.3f} val= {vloss:.3f}")
    return model, history

def nn_dataset(dt, key_1d, key_2d):
  dt_1dx = np.array(dt[key_1d])
  dt_1dx = np.reshape(dt_1dx,(-1,len(key_1d),1))
  dt_1dx = torch.tensor(dt_1dx, dtype=torch.float32)
  
  dt_2dx = np.array(dt[key_2d])
  dt_2dx = np.reshape(dt_2dx,(-1,int(len(key_2d)/2),2))
  dt_2dx = torch.tensor(dt_2dx, dtype=torch.float32)
  dt_y = dt[pref["pred"]].tolist() 
  return dt_1dx, dt_2dx, dt_y
  
class CustomDataset(Dataset):
  def __init__(self, x1d, x2d, y):       
      self.x1d = x1d
      self.x2d = x2d
      self.y = torch.tensor(y, dtype=torch.float32)

  def __len__(self):
      return len(self.y)

  def __getitem__(self, idx):
      return self.x1d[idx], self.x2d[idx], self.y[idx]
        
class NN_Model(nn.Module):
  def __init__(self, num_1d, num_2d):
    super(NN_Model, self).__init__()
    self.x1d_mean = None
    self.x1d_std = None
    self.x2d_mean = None
    self.x2d_std = None
    self.num_1d = num_1d
    self.num_2d = num_2d
    
    self.nn_1d = nn.ModuleList([
        nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        ) for _i in range(self.num_1d)])
    self.nn_2d = nn.ModuleList([
        nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        ) for _i in range(self.num_2d)])
    self.fc = nn.Sequential(  
        nn.Linear((self.num_1d + self.num_2d) * 16, 128),
        nn.ReLU(),
        nn.Linear(128, 1))    

  def forward(self, x1d, x2d):  
    x2d, x1d = self.normalize(x1d, x2d)
           
    data1d = [data(x1d[:, i:i+1]) for i, data in enumerate(self.nn_1d)]
    data1d = torch.cat(data1d, dim=1)          
    data2d = [data(x2d[:, i:i+1]) for i, data in enumerate(self.nn_2d)]
    data2d = torch.cat(data2d, dim=1)

    combined = torch.cat((data1d, data2d), dim=1)  
    combined = combined.view(combined.size(0), -1)

    return self.fc(combined)
      
  def normalize(self, x1d, x2d):
    x1d_normalized = (x1d - self.x1d_mean) / self.x1d_std
    x2d_normalized = (x2d - self.x2d_mean) / self.x2d_std
    return x2d_normalized, x1d_normalized     
      
  def set_normalization_params(self, x1d, x2d):
    self.x1d_mean = x1d.sum() / x1d.size(0)
    self.x1d_std = ((x1d - self.x1d_mean).pow(2).sum() / x1d.size(0)).sqrt()
    x2d_rs = x2d.reshape(-1, x2d.size(-1))
    xsize = x2d_rs.size(0)
    self.x2d_mean = x2d_rs.sum() / xsize
    self.x2d_std = ((x2d_rs - self.x2d_mean).pow(2).sum() / xsize).sqrt()          

def prepare_nn_test(feature, key_1d, key_2d):
  test_set, dt_pred = prepare_test(feature)
  test_set[pref['pred']] = dt_pred[pref['pred']]
  test_1dx, test_2dx, test_y = nn_dataset(test_set, key_1d, key_2d)
  test_dataset = CustomDataset(test_1dx, test_2dx, test_y)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=pref['batch'], shuffle=True)
   
  return test_loader
      
def nn_feature(feature):
  key_1d, key_2d = [], []
  key_E = [[i for i in feature if 'E_' in i]]
  for k in ['ion_','hb_','E_']:
    key_1d.extend([i for i in feature if k in i])
  num_1d = len(key_1d)  
  for k in ['dih','cp1','cp2']:
    key_2d.extend([i for i in feature if k in i])
  num_2d = int(len(key_2d)/2)  
  return key_1d, key_2d, num_1d, num_2d
    
def nn_train():      
  ### prepare train and test set
  train_set, feature = prepare_dataset(pref["train"])  
  key_1d, key_2d, num_1d, num_2d = nn_feature(feature)
    
  train_1dx, train_2dx, train_y = nn_dataset(train_set, key_1d, key_2d)
  train_dataset = CustomDataset(train_1dx, train_2dx, train_y)     
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=pref['batch'], shuffle=True)
  test_loader = prepare_nn_test(feature, key_1d, key_2d)
    
  model = NN_Model(num_1d,num_2d)
  model.set_normalization_params(train_1dx, train_2dx)
  prop_stat = [model.x1d_mean.item(),
               model.x1d_std.item(),
               model.x2d_mean.item(),
               model.x2d_std.item()]
  
  trained_model, history = fit_model(model, train_loader, test_loader, epochs=pref['epochs'], lr=pref['lr'])
   
  torch.save({
           'state_dict': trained_model.state_dict(),
           'feature': feature,
           'prop_stat':prop_stat},
           f"nn_{pref['pred_fn']}.pth")
  
  nn_evaluate(trained_model, test_loader)
    
def nn_predict(test_loader, trained_model):    
  pred = []
  real = []
  with torch.no_grad():    
    for x1d_batch, x2d_batch, y_batch in test_loader:
      x1d_batch = x1d_batch.to(pref['device'])
      x2d_batch = x2d_batch.to(pref['device']) 
      real.extend(y_batch.flatten().tolist())
      
      predictions = trained_model(x1d_batch, x2d_batch)
      pred.extend(predictions.flatten().tolist())
  return pred, real    
  
def nn_evaluate(chk, test_loader=[]): 
  if not test_loader:
    key_1d, key_2d, num_1d, num_2d = nn_feature(chk['feature'])
    test_loader = prepare_nn_test(chk['feature'], key_1d, key_2d)
    print (f"expended feature= \n {chk['feature']}\n")
        
    model = nn_load_model(chk, num_1d,num_2d)
    model.eval() 
  else:
    model = chk
    
  pred, real = nn_predict(test_loader, model)   
  dt_pred = pd.DataFrame()    
  pred_l = [ int(i < pref['target'][1]) for i in pred]
  real_l = [ int(i < pref['target'][1]) for i in real]
  dt_pred['pred'] = pred
  dt_pred['pred_l'] = pred_l
  dt_pred['act'] = real
  dt_pred['E'] = real
  dt_pred['act_l'] = real_l
  
  dt_pred.to_csv(f"pred_{pref['pred_fn']}.csv", index=False)
  
  best_model = label_evaluate(dt_pred, 0)
  hist_fig(dt_pred)
  
def nn_load_model(chk, num_1d, num_2d):
  model = NN_Model(num_1d,num_2d)
  model.load_state_dict(chk['state_dict'])
  model.x1d_mean = chk['prop_stat'][0]
  model.x1d_std = chk['prop_stat'][1]
  model.x2d_mean = chk['prop_stat'][2]
  model.x2d_std = chk['prop_stat'][3]
  model.to(pref['device'])
  return model
   
####################################
####################################

if __name__ == "__main__":
  pref = preprocess()
    
  if pref['run'] == 'train':
    if pref['ml'] == 'rf':
      rf_train()
    elif pref['ml'] == 'nn':
      nn_train()
  elif pref['run'] == 'evaluate': 
    if pref['ml'] == 'rf':  
      model = joblib.load(pref["model"])
      rf_evaluate(model)  
    elif pref['ml'] == 'nn':
      chkpt = torch.load(pref["model"])
      nn_evaluate(chkpt)
               
  elif pref['run'] == 'predict': 
    if pref['ml'] == 'rf':  
      model = joblib.load(pref["model"])
      rf_evaluate(model)  
    elif pref['ml'] == 'nn':
      print (456)    
      

   