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

def defense_score(use_trimmed=False, lower_percentile=5, upper_percentile=95):
    """
    Calculate defense scores with optional trimmed normalization.

    Parameters:
    -----------
    use_trimmed : bool, default=False
        If True, uses trimmed min-max normalization to exclude outliers.
        If False, uses standard min-max normalization from the entire distribution.

        Use trimmed normalization when:
        - Extreme outliers are skewing the score distribution
        - You want to prevent a few exceptional cases from compressing the majority
        - You need better differentiation among most students

    lower_percentile : float, default=5
        Lower percentile for trimmed normalization (e.g., 5 = bottom 5%).
        Only applies when use_trimmed=True.
        Adjustable based on data distribution (e.g., 1, 5, 10).

    upper_percentile : float, default=95
        Upper percentile for trimmed normalization (e.g., 95 = top 5%).
        Only applies when use_trimmed=True.
        Adjustable based on data distribution (e.g., 90, 95, 99).
    """
    # Step 1: Collect all original defense scores
    all_am = []
    for j in range(num_all_id):
        am = np.load('mark/'+str(j)+'.npy')
        am = am[am != 0]
        ori = np.mean(am)
        all_am.append(ori)

    all_am = np.array(all_am)

    # Step 2: Determine min/max based on normalization method
    if use_trimmed:
        # Trimmed normalization: use percentiles to exclude outliers
        min_score = np.percentile(all_am, lower_percentile)
        max_score = np.percentile(all_am, upper_percentile)
    else:
        # Standard normalization: use actual min/max from the distribution
        min_score = np.min(all_am)
        max_score = np.max(all_am)

    # Step 3: Normalize each student's score
    for j in range(num_all_id):
        am = np.load('mark/'+str(j)+'.npy')
        am = am[am != 0]
        ori = np.mean(am)

        # Calculate normalized score
        if max_score - min_score > 0:
            nor = (ori - min_score) / (max_score - min_score)
            if use_trimmed:
                nor = np.clip(nor, 0, 1)  # Clip to [0, 1] when using trimmed method
        else:
            nor = 0.5  # If all scores are the same

        f = 'grade/'+str(rn_id(j))+'.txt'
        with open(f,"a") as file:
            file.write('original defense score: '+str(ori)+'  (0-1)normalized defense score: '+str(nor)+"\n")

def attack_score(use_trimmed=False, lower_percentile=5, upper_percentile=95):
    """
    Calculate attack scores with optional trimmed normalization.

    Parameters:
    -----------
    use_trimmed : bool, default=False
        If True, uses trimmed min-max normalization to exclude outliers.
        If False, uses standard min-max normalization from the entire distribution.

        Use trimmed normalization when:
        - Extreme outliers are skewing the score distribution
        - You want to prevent a few exceptional cases from compressing the majority
        - You need better differentiation among most students

    lower_percentile : float, default=5
        Lower percentile for trimmed normalization (e.g., 5 = bottom 5%).
        Only applies when use_trimmed=True.
        Adjustable based on data distribution (e.g., 1, 5, 10).

    upper_percentile : float, default=95
        Upper percentile for trimmed normalization (e.g., 95 = top 5%).
        Only applies when use_trimmed=True.
        Adjustable based on data distribution (e.g., 90, 95, 99).
    """
    # Step 1: Build attack matrix
    all_am = np.zeros((num_all_id,num_all_id))
    for j in range(num_all_id):
        am = np.load('mark/'+str(j)+'.npy')
        all_am[j] = am
    all_am = all_am.T

    # Step 2: Collect all original attack scores
    all_am_ = []
    for i in range(num_all_id):
        aa = all_am[i][all_am[i] != 0]
        ori = 1/np.mean(aa)
        all_am_.append(ori)

    all_am_ = np.array(all_am_)

    # Step 3: Determine min/max based on normalization method
    if use_trimmed:
        # Trimmed normalization: use percentiles to exclude outliers
        min_score = np.percentile(all_am_, lower_percentile)
        max_score = np.percentile(all_am_, upper_percentile)
    else:
        # Standard normalization: use actual min/max from the distribution
        min_score = np.min(all_am_)
        max_score = np.max(all_am_)

    # Step 4: Normalize each student's score
    for i in range(num_all_id):
        aa = all_am[i][all_am[i] != 0]
        ori = 1/np.mean(aa)

        # Calculate normalized score
        if max_score - min_score > 0:
            nor = (ori - min_score) / (max_score - min_score)
            if use_trimmed:
                nor = np.clip(nor, 0, 1)  # Clip to [0, 1] when using trimmed method
        else:
            nor = 0.5  # If all scores are the same

        f = 'grade/'+str(rn_id(i))+'.txt'
        with open(f,"a") as file:
            file.write('original attack score: '+str(ori)+'  (0-1)normalized attack score: '+str(nor)+"\n") 
