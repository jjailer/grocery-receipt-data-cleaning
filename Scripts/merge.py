import itertools

from tqdm.auto import tqdm
import pandas as pd


def equalize_length(df1, df2):
    df1_length, df2_length = len(df1), len(df2)
    
    if df1_length > df2_length:    
        df2 = df2.reindex(list(range(df1_length))).fillna('')
    elif df2_length > df1_length:
        df1 = df1.reindex(list(range(df2_length))).fillna('')

    assert len(df1) == len(df2)
    return df1, df2


def compare_naive(df_row):
    result = []
    df_split = df_row.str.split()
    df_split = df_split.fillna('') # hack: somehow a NaN value can sneak in 
    
    shared_words = set.intersection(*map(set, df_split))
    unshared_words = set.symmetric_difference(*map(set, df_split))
    
    shared_subwords = []
    subword_matches = []
    for word1 in unshared_words:
        for word2 in unshared_words:
            if word1 in word2 and word1 != word2:
                shared_subwords.append(word1)
                subword_matches.extend([word1, word2])
    unmatched_words = unshared_words.difference(set(subword_matches))
    
    unmatched_words_in_voc = [word for word in unmatched_words if word in word_vectors.vocab]     
    shared_words_in_voc = [word for word in shared_words if word in word_vectors.vocab]
    unshared_words_in_voc = [word for word in unshared_words if word in word_vectors.vocab]
    words_in_voc = shared_words_in_voc + unshared_words_in_voc
    
    # Matching Case Flow
    # Return values are small identifiable numbers, primarily used as flags, but their ordering is important
    # 1. complete identical match
    # 2. one contains all the words of the other
    # 3. one contains all the words of the other as substrings
    # 4. one unmatched word
    # 5. unmatched words in vocabulary
    # 6. unmatched words but out of vocabulary
    
    # always return identical matches
    result.extend(shared_words)
    
    # 1. complete identical match
    if not unshared_words:  
        df_row['WordVec'] = ' '.join(result)
        df_row['Distance'] = 0
        return df_row
    
    # 2. one contains all the words of the other
    if set(df_split[0]) == shared_words or set(df_split[1]) == shared_words:  
        df_row['WordVec'] = ' '.join(result)
        df_row['Distance'] = 0.123456  
        return df_row
    
    # 3. one contains all the words of the other as substrings
    result.extend(shared_subwords)
    if not unmatched_words:   
        df_row['WordVec'] = ' '.join(result)
        df_row['Distance'] = 0.271828  
        return df_row            
    # 4. one unmatched word
    elif len(unmatched_words) == 1:
        result.append(*unmatched_words)
        df_row['WordVec'] = ' '.join(result)
        df_row['Distance'] = 0.314159       
        return df_row
  
    # 5. unmatched words in vocabulary
    if unmatched_words_in_voc:
        most_similar_key, _ = word_vectors.most_similar(positive=[*shared_words_in_voc, *unmatched_words_in_voc], topn=1)[0]  
        
        # don't append word vector if it repeats a word or subword
        for res in result: 
            if most_similar_key in res or res in most_similar_key:
                df_row['WordVec'] = ' '.join(result)
                df_row['Distance'] = 0.666
                return df_row
            
        result.append(most_similar_key)
        df_row['WordVec'] = ' '.join(result)
        df_row['Distance'] = word_vectors.wmdistance(df_split[0], df_split[1]) # wmdistance handles oov
        return df_row
    # 6. unmatched words but out of vocabulary
    else:
        df_row['WordVec'] = ' '.join(result)
        df_row['Distance'] = 0.999  


def align(df1, df2, count_only=False):
    df1, df2 = equalize_length(df1, df2)
    
    # remove identical matches
    result_pairs = []
    df1_dropped, df2_dropped = df1.index, df2.index
    for df1_idx, df1_word in df1.iteritems():
        matches = df2[df2_dropped].str.fullmatch(df1_word)
        if any(matches):
            match_index = matches.idxmax() # return index of first match
            result_pairs.append((df1_idx, match_index))
            df1_dropped = df1_dropped.drop(df1_idx)
            df2_dropped = df2_dropped.drop(match_index)
    
    # remove substring matches
    df1_split = df1[df1_dropped].str.split()
    for df1_idx, df1_words in df1_split.iteritems():
        for word in df1_words:
            matches = df2[df2_dropped].str.contains(word, regex=False)
            if any(matches):
                match_index = matches.idxmax() # return index of first match
                result_pairs.append((df1_idx, match_index))
                df1_dropped = df1_dropped.drop(df1_idx)
                df2_dropped = df2_dropped.drop(match_index)
                break
    
    # remove substring matches in the other direction
    df2_split = df2[df2_dropped].str.split()
    for df2_idx, df2_words in df2_split.iteritems():
        for word in df2_words:
            matches = df1[df1_dropped].str.contains(word, regex=False)
            if any(matches):
                match_index = matches.idxmax() # return index of first match
                result_pairs.append((match_index, df2_idx))
                df1_dropped = df1_dropped.drop(match_index)
                df2_dropped = df2_dropped.drop(df2_idx)
                break
    
    # remove additional unmatched empty items
    df2_dropped = df2_dropped.drop(df2[df2_dropped][df2[df2_dropped] == ''].index)    

    ## short circut for debugging permutation counts
    if count_only:
        print(f'{len(df2_dropped)}! {list(df2_dropped)}')
        return
    
    # all permutations of remaining indices
    perms = itertools.permutations(df2_dropped)

    # generate word vectors and similarity
    if len(df2_dropped) > 1:
        total_distance = []
        df1_reindexed = df1[df1_dropped].reset_index(drop=True)
        for p in tqdm(perms, desc="Permutations", leave=False):
            p = pd.Index(p)
            total_distance.append(
                sum(pd.concat(
                    [df1_reindexed, df2[p].reset_index(drop=True)], axis=1).apply(compare_naive, axis=1).Distance))        
        # find max permutation
        perms_reset = itertools.permutations(df2_dropped) # reset generator
        result_index = pd.Index(next(itertools.islice(perms_reset, total_distance.index(min(total_distance)), None)))
    else:
        perms_reset = itertools.permutations(df2_dropped) # reset generator
        result_index = pd.Index(next(itertools.islice(perms_reset, 0, None)))
    
    # return concatendated dataframe with word vectors
    if result_pairs:
        top_index_left, top_index_right = map(pd.Index, zip(*result_pairs))
    else:
        top_index_left, top_index_right = pd.Index([]), pd.Index([])
    bot_index_left, bot_index_right = df1_dropped, result_index

    df_combined = pd.concat([pd.concat([df1[top_index_left].reset_index(drop=True), 
                                        df2[top_index_right].reset_index(drop=True)], axis=1, ignore_index=True), 
                             pd.concat([df1[bot_index_left].reset_index(drop=True), 
                                        df2[bot_index_right].reset_index(drop=True)], axis=1, ignore_index=True)], 
                            ignore_index=True)
    
    return df_combined


def align_by_receipt(dfs, wv, count_only=False):
    # align([dfs], word_vectors, comparison_function, count_only, dfwv)
    # return(df, [dfwv])
    global word_vectors 
    word_vectors = wv
    df_final = pd.DataFrame()
    IDs = [130, 153, 135, 137, 141, 114, 121, 127]
    for pid in tqdm(IDs, desc="IDs"):
        for session in tqdm(dfs[0].loc[dfs[0].ID == pid, 'Session'].unique(), desc="Sessions"):
            for receipt in tqdm(dfs[0].loc[(dfs[0].ID == pid) & (dfs[0].Session == session), 'Receipt'].unique(), desc="Receipts"):
                print(f'ID: {pid}, Session: {session}, Receipt: {receipt}')
                if count_only:
                    df_final = align(dfs[0].loc[(dfs[0].ID == pid) & 
                                                (dfs[0].Session == session) & 
                                                (dfs[0].Receipt == receipt), 'Item'].reset_index(drop=True), 
                                     dfs[2].loc[(dfs[2].ID == pid) & 
                                                (dfs[2].Session == session) & 
                                                (dfs[2].Receipt == receipt), 'Item'].reset_index(drop=True),
                                     count_only)
                else:
                    df_final = pd.concat([df_final, align(dfs[0].loc[(dfs[0].ID == pid) & 
                                                                     (dfs[0].Session == session) & 
                                                                     (dfs[0].Receipt == receipt), 'Item'].reset_index(drop=True),
                                                          dfs[2].loc[(dfs[2].ID == pid) & 
                                                                     (dfs[2].Session == session) & 
                                                                     (dfs[2].Receipt == receipt), 'Item'].reset_index(drop=True), 
                                                          count_only).apply(compare_naive, axis=1)],
                                         ignore_index=True)
    return df_final