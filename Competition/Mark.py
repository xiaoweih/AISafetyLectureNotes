def evaluate_all_models(id_):
    a_mark = np.zeros((num_all_id))
    model = Net().to(device)

    model.load_state_dict(torch.load(path_id))
    for i in range(num_all_id):
        adv_attack = rt_attack(i)
        ls, acc = eval_adv_test(model, device, test_loader, adv_attack)
        a_mark[i] = acc
    np.save('mark/'+str(id_)+".npy",a_mark)        
    del model

def defense_score():
    all_am = []
    for j in range(num_all_id):
        am = np.load('mark/'+str(j)+'.npy')
        am = am[am != 0]
        ori = np.mean(am)
        #smallest original defense score: 0.07036072530864196, largest original defense score: 0.7646020683453237
        nor = (ori-0.07036072530864196)/(0.7646020683453237-0.07036072530864196)
        f = 'grade/'+str(rn_id(j))+'.txt'
        with open(f,"a") as file:
            file.write('original defense score: '+str(ori)+'  (0-1)normalized defense score: '+str(nor)+"\n")

def attack_score():
    all_am = np.zeros((num_all_id,num_all_id))
    for j in range(num_all_id):
        am = np.load('mark/'+str(j)+'.npy')
        all_am[j] = am
    all_am = all_am.T

    all_am_ = []
    for i in range(num_all_id):
        aa = all_am[i][all_am[i] != 0]
        ori = 1/np.mean(aa)
        #smallest original attack score: 1.0963935013604147, largest original attack score: 3.9667957941339234
        nor = (ori-1.0963935013604147)/(3.9667957941339234-1.0963935013604147)
        f = 'grade/'+str(rn_id(i))+'.txt'
        with open(f,"a") as file:
            file.write('original attack score: '+str(ori)+'  (0-1)normalized attack score: '+str(nor)+"\n") 
