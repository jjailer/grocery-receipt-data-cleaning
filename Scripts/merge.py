import itertools as it

from tqdm.notebook import tqdm
import pandas as pd


def _equalize_length(df1, df2):
    """Extend the length of one DataFrame to match the length of the other.
    
    Additional rows contain the empty string.
    Return the (initially) larger DataFrame first. """
    # df1 must point to larger df
    # align function searches for matches from df1 into df2
    # filling with the empty string in df1 would cause surpious matches 
    if df1.shape[0] == df2.shape[0]:
        return df1, df2
    elif df2.shape[0] > df1.shape[0]:
        df1, df2 = df2, df1
    
    df2 = df2.reindex(list(range(df1.shape[0])))
    df2.ID = df2.ID.fillna(method='ffill')
    df2.Session = df2.Session.fillna(method='ffill')
    df2.Receipt = df2.Receipt.fillna(method='ffill')
    df2.Item = df2.Item.fillna('')

    assert df1.shape[0] == df2.shape[0]
    return df1, df2


def compare_naive(df_row, word_vectors):
    result = []
    df_split = df_row.str.split()
    # df_split = df_split.fillna('') # hack: somehow a NaN value can sneak in 
    
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
    
    shared_words_in_voc = [word for word in shared_words if word in word_vectors.vocab]
    unshared_words_in_voc = [word for word in unshared_words if word in word_vectors.vocab]
    unmatched_words_in_voc = [word for word in unmatched_words if word in word_vectors.vocab]     
    # words_in_voc = [word for item in df_split for word in item if word in word_vectors.vocab]
    words_in_voc = shared_words_in_voc + unshared_words_in_voc
    
    # Matching Case Flow
    # Return values are small identifiable numbers, primarily used as flags, but their ordering is important
    # 1. complete identical match
    # 2. one contains all the words of the other
    # 3. one contains all the words of the other as substrings
    # 4. result contains two or more words
    # 5. one unmatched word in vocabulary
    # 6. multiple unmatched words in vocabulary
    # 7. unmatched words but out of vocabulary
    
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
        df_row['Distance'] = 0.16180
        return df_row    
    
    # 4. result contains two or more words
    if len(result) >= 2:
        df_row['WordVec'] = ' '.join(result)
        df_row['Distance'] = 0.271828  
        return df_row    
    
    # 5. one unmatched word in vocabulary
    if len(unmatched_words_in_voc) == 1:
        result.append(*unmatched_words_in_voc)
        df_row['WordVec'] = ' '.join(result)
        df_row['Distance'] = 0.314159       
        return df_row
    # 6. multiple unmatched words in vocabulary
    elif len(unmatched_words_in_voc) > 1:
        most_similar_key, _ = word_vectors.most_similar(positive=[*words_in_voc], topn=1)[0]  
        
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
    # 7. unmatched words but out of vocabulary
    else:
        df_row['WordVec'] = ' '.join(result)
        df_row['Distance'] = 0.999  
        return df_row


def align(df1, df2, word_vectors, count_only=False):
    df1, df2 = _equalize_length(df1, df2)
    
    # remove identical matches
    result_pairs = []
    df1_dropped, df2_dropped = df1.index, df2.index
    for df1_idx, df1_word in df1.Item.iteritems():
        matches = df2.loc[df2_dropped, 'Item'].str.fullmatch(df1_word, na=False)
        if any(matches):
            match_index = matches.idxmax() # return index of first match
            result_pairs.append((df1_idx, match_index))
            df1_dropped = df1_dropped.drop(df1_idx)
            df2_dropped = df2_dropped.drop(match_index)
    
    # remove substring matches
    df1_split = df1.loc[df1_dropped, 'Item'].str.split()
    for df1_idx, df1_words in df1_split.iteritems():
        for word in df1_words:
            matches = df2.loc[df2_dropped, 'Item'].str.contains(word, regex=False)
            if any(matches):
                match_index = matches.idxmax() # return index of first match
                result_pairs.append((df1_idx, match_index))
                df1_dropped = df1_dropped.drop(df1_idx)
                df2_dropped = df2_dropped.drop(match_index)
                break
    
    # remove substring matches in the other direction
    df2_split = df2.loc[df2_dropped, 'Item'].str.split()
    for df2_idx, df2_words in df2_split.iteritems():
        for word in df2_words:
            matches = df1.loc[df1_dropped, 'Item'].str.contains(word, regex=False)
            if any(matches):
                match_index = matches.idxmax() # return index of first match
                result_pairs.append((match_index, df2_idx))
                df1_dropped = df1_dropped.drop(match_index)
                df2_dropped = df2_dropped.drop(df2_idx)
                break
    
    # remove additional unmatched empty items
    df2_dropped = df2_dropped.drop(df2.loc[df2_dropped.intersection(df2.Item == '')].index)    

    ## short circut for debugging permutation counts (divergence function)
    if count_only:
        print(f'{len(df2_dropped)}! {list(df2_dropped)}')
        return
    
    # all permutations of remaining indices
    perms = it.permutations(df2_dropped)

    # generate word vectors and similarity
    if len(df2_dropped) > 1:
        total_distance = []
        df1_reindexed = df1.loc[df1_dropped, :].reset_index(drop=True)
        for p in perms:
            p = pd.Index(p)
            total_distance.append(
                sum(pd.concat(
                    [df1_reindexed.Item, df2.loc[p, 'Item'].reset_index(drop=True)], axis=1).apply(compare_naive, 
                                                                                  axis=1, args=(word_vectors,)).Distance))        
        # find max permutation
        perms_reset = it.permutations(df2_dropped) # reset generator
        result_index = pd.Index(next(it.islice(perms_reset, total_distance.index(min(total_distance)), None)))
    else:
        perms_reset = it.permutations(df2_dropped) # reset generator
        result_index = pd.Index(next(it.islice(perms_reset, 0, None)))
    
    # return concatendated dataframe with word vectors
    if result_pairs:
        top_index_left, top_index_right = map(pd.Index, zip(*result_pairs))
    else:
        top_index_left, top_index_right = pd.Index([]), pd.Index([])
    bot_index_left, bot_index_right = df1_dropped, result_index

    df_combined = pd.concat([pd.concat([df1.loc[top_index_left, :].reset_index(drop=True), 
                                        df2.loc[top_index_right, :].reset_index(drop=True)], axis=1, ignore_index=True), 
                             pd.concat([df1.loc[bot_index_left, :].reset_index(drop=True), 
                                        df2.loc[bot_index_right, :].reset_index(drop=True)], axis=1, ignore_index=True)], 
                            ignore_index=True)
    
    return df_combined


def divergence(dfs, word_vectors):
    if len(dfs) != 2:
        raise TypeError("Expected list of length 2")
    
    for df in dfs:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame")
        if df.columns.values.tolist() != ["ID", "Session", "Receipt", "Item"]:
            raise ValueError('Expected columns "ID", "Session", "Receipt", "Item"')
        df.Item.fillna('')
        
    if set(dfs[0].ID.unique()) != set(dfs[1].ID.unique()):
        raise ValueError("Expected identical ID values")
    if set(dfs[0].Session.unique()) != set(dfs[1].Session.unique()):
        raise ValueError("Expected identical Session values")
    if set(dfs[0].Receipt.unique()) != set(dfs[1].Receipt.unique()):
        raise ValueError("Expected identical Receipt values")
        
    df_final = pd.DataFrame()
    
    for pid in dfs[0].ID.unique():
        for session in dfs[0].loc[dfs[0].ID == pid, 'Session'].unique():
            for receipt in dfs[0].loc[(dfs[0].ID == pid) & (dfs[0].Session == session), 'Receipt'].unique():
                print(f'ID: {pid}, Session: {session}, Receipt: {receipt}, Div:', end=' ')
                align(dfs[0].loc[(dfs[0].ID == pid) & (dfs[0].Session == session) & 
                                 (dfs[0].Receipt == receipt)].reset_index(drop=True), 
                      dfs[1].loc[(dfs[1].ID == pid) & (dfs[1].Session == session) & 
                                 (dfs[1].Receipt == receipt)].reset_index(drop=True), 
                      word_vectors, count_only=True)
        print()
    return


def merge(dfs, word_vectors):
    if len(dfs) != 2:
        raise TypeError("Expected list of length 2")
    
    for df in dfs:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame")
        if df.columns.values.tolist() != ["ID", "Session", "Receipt", "Item"]:
            raise ValueError('Expected columns "ID", "Session", "Receipt", "Item"')
        df.Item.fillna('')
        
    if set(dfs[0].ID.unique()) != set(dfs[1].ID.unique()):
        raise ValueError("Expected identical ID values")
    if set(dfs[0].Session.unique()) != set(dfs[1].Session.unique()):
        raise ValueError("Expected identical Session values")
    if set(dfs[0].Receipt.unique()) != set(dfs[1].Receipt.unique()):
        raise ValueError("Expected identical Receipt values")
        
    df_large = pd.DataFrame()
    
    for pid in tqdm(dfs[0].ID.unique(), desc="ID"):
        for session in tqdm(dfs[0].loc[dfs[0].ID == pid, 'Session'].unique(), desc="Session"):
            for receipt in dfs[0].loc[(dfs[0].ID == pid) & (dfs[0].Session == session), 'Receipt'].unique():
                df_large = pd.concat([df_large, align(dfs[0].loc[(dfs[0].ID == pid) & 
                                                                 (dfs[0].Session == session) & 
                                                                 (dfs[0].Receipt == receipt)].reset_index(drop=True),
                                                      dfs[1].loc[(dfs[1].ID == pid) & 
                                                                 (dfs[1].Session == session) & 
                                                                 (dfs[1].Receipt == receipt)].reset_index(drop=True), 
                                                      word_vectors)],
                                     ignore_index=True)
    
    df_large = df_large.rename({0: 'ID', 1: 'Session', 2: 'Receipt'}, axis=1)
    df_items = pd.DataFrame(df_large.iloc[:, [3, 7]])
    df_items.columns = ['0', '1']
    
    df_final = pd.concat([df_large.iloc[:, [0, 1, 2]], df_items.apply(compare_naive, axis=1, args=(word_vectors,))], axis=1)
    df_final = df_final.rename({'0': 'Item1', '1': 'Item2', 'WordVec': 'Item'}, axis=1)
    return df_final[['ID', 'Session', 'Receipt', 'Item']], df_final.loc[df_final.Distance > 1]