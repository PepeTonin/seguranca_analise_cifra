import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'key': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'],
    'plain_text': ['a', 'aa', 'aaa', 'aaaa', 'aaaaa', 'aaaaaa', 'aaaaaaa', 'aaaaaaaa', 'aaaaaaaaa', 'aaaaaaaaaa',
                   'aaaaaaaaaaa', 'aaaaaaaaaaaa', 'aaaaaaaaaaaaa', 'aaaaaaaaaaaaaa', 'aaaaaaaaaaaaaaa',
                   'aaaaaaaaaaaaaaaa', 'aaaaaaaaaaaaaaaaa', 'aaaaaaaaaaaaaaaaaa'],
    'enc_text': ['OKJLSQJ', 'YUTVCAYUUVCAT', 'KGGHOMKGHHOMKGFHOMF', 'OKKLSQOKLLSQOKJLSQOKMLSQJ',
                 'GCBDKIGCEDKIGCCDKIGCDDKIGCFDKIB', 'HDGELJHDCELJHDFELJHDEELJHDDELJHDHELJC',
                 'NJLKRPNJKKRPNJIKRPNJJKRPNJNKRPNJMKRPNJOKRPI', 'PLMMTRPLKMTRPLLMTRPLNMTRPLOMTRPLRMTRPLPMTRPLQMTRK',
                 'BXAYFDBXYYFDBXZYFDBXDYFDBXWYFDBXCYFDBXEYFDBXXYFDBXBYFDW',
                 'ZVAWDBZVYWDBZVCWDBZVUWDBZVZWDBZVWWDBZVXWDBZVDWDBZVVWDBZVBWDBU',
                 'JFHGNLJFGGNLJFIGNLJFNGNLJFLGNLKGFEGNLJFKGNLJFFGNLJFJGNLJFEGNLJFMGNLE',
                 'ZVWWDBZVYWDBZVAWDBZVBWDBZVZWDBZVVWDBZVUWDBAWVVWDBAWVUWDBZVDWDBZVCWDBZVXWDBU',
                 'TPRQXVTPQQXVTPSQXVTPTQXVTPWQXVTPVQXVTPXQXVUQPPQXVUQPOQXVUQPQQXVTPOQXVTPPQXVTPUQXVO',
                 'DZDAHFEAZZAHFDZBAHFDZFAHFDZCAHFEAZAAHFDZAAHFEAZYAHFDZGAHFDZEAHFDZHAHFEAZBAHFDZZAHFDZYAHFY',
                 'UQQRYWUQPRYWVRQSRYWUQWRYWVRQRRYWUQRRYWVRQPRYWUQVRYWUQSRYWUQURYWUQTRYWUQXRYWVRQQRYWVRQTRYWUQYRYWP',
                 'TPQQXVUQPOQXVUQPRQXVTPXQXVUQPQQXVTPUQXVUQPPQXVTPPQXVTPTQXVTPSQXVTPRQXVUQPSQXVUQPTQXVTPOQXVTPWQXVTPVQXVO',
                 'BXXYFDCYXBYFDCYXYYFDBXBYFDCYXCYFDCYXXYFDBXAYFDBXFYFDCYXWYFDBXEYFDBXWYFDBXYYFDCYXZYFDCYXAYFDBXCYFDBXZYFDBXDYFDW',
                 'YUXVCAZVUXVCAYUBVCAYUUVCAZVUYVCAYUYVCAYUZVCAZVUVVCAYUTVCAZVUUVCAYUVVCAYUAVCAYUCVCAZVUTVCAYUWVCAZVUZVCAZVUWVCAZVUAVCAT']
}

data_to_concat_1 = {
    'key': ['aa', 'aa', 'aa', 'aa', 'aa', 'aa', 'aa', 'aa', 'aa', 'aa', 'aa'],
    'plain_text': ['a', 'aa', 'aaa', 'aaaa', 'aaaaa', 'aaaaaa', 'aaaaaaa', 'aaaaaaaa', 'aaaaaaaaa', 'aaaaaaaaaa',
                   'aaaaaaaaaaa'],
    'enc_text': ['UQPRYWP', 'RNNOVTRNMOVTM', 'XTUUBZXTSUBZXTTUBZS', 'YUUVCAYUTVCAYUVVCAYUWVCAT',
                 'NJKKRPNJMKRPNJIKRPNJLKRPNJJKRPI', 'ZVVWDBZVWWDBZVUWDBZVYWDBZVXWDBZVZWDBU',
                 'LHIIPNLHKIPNLHLIPNLHGIPNLHHIPNLHMIPNLHJIPNG', 'OKKLSQOKNLSQOKMLSQOKJLSQOKOLSQOKPLSQOKQLSQOKLLSQJ',
                 'QMPNUSQMQNUSQMTNUSQMLNUSQMSNUSQMONUSQMRNUSQMMNUSQMNNUSL',
                 'NJJKRPNJPKRPNJOKRPNJNKRPNJMKRPNJLKRPNJKKRPNJIKRPNJQKRPNJRKRPI',
                 'BXBYFDBXEYFDBXDYFDBXZYFDBXYYFDBXFYFDBXCYFDBXWYFDBXAYFDBXXYFDCYXWYFDW']
}

data_to_concat_2 = {
    'key': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'],
    'plain_text': ['b', 'bb', 'bbb', 'bbbb', 'bbbbb', 'bbbbbb', 'bbbbbbb', 'bbbbbbbb', 'bbbbbbbbb', 'bbbbbbbbbb'],
    'enc_text': ['BXWYFEW', 'NJIKRQNJJKRQI', 'LHGIPOLHIIPOLHHIPOG', 'DZZAHGDZAAHGDZBAHGDZYAHGY',
                 'MIKJQPMIJJQPMILJQPMIIJQPMIHJQPH', 'VRTSZYVRUSZYVRSSZYVRRSZYVRQSZYVRVSZYQ',
                 'QMNNUTQMONUTQMPNUTQMLNUTQMQNUTQMRNUTQMMNUTL', 'NJJKRQNJLKRQNJPKRQNJNKRQNJOKRQNJIKRQNJMKRQNJKKRQI',
                 'NJJKRQNJPKRQNJLKRQNJOKRQNJQKRQNJIKRQNJNKRQNJMKRQNJKKRQI',
                 'QMLNUTQMSNUTQMPNUTQMUNUTQMRNUTQMMNUTQMTNUTQMONUTQMNNUTQMQNUTL']
}


def is_all_arrays_same_length(data):
    result = True
    key_length = len(data['key'])
    plain_text_length = len(data['plain_text'])
    enc_text_length = len(data['enc_text'])
    if ((key_length != plain_text_length) or (key_length != enc_text_length) or (plain_text_length != enc_text_length)):
        result = False
    return result


def verify_all_data(array_data):
    for data in array_data:
        if not (is_all_arrays_same_length(data)):
            return False
    return True


is_all_data_approved = verify_all_data([data, data_to_concat_1, data_to_concat_2])

if not is_all_data_approved:
    print('Erro na análise dos dados')
else:
    df = pd.DataFrame(data)
    df_to_concat_1 = pd.DataFrame(data_to_concat_1)
    df_to_concat_2 = pd.DataFrame(data_to_concat_2)
    df = pd.concat([df, df_to_concat_1, df_to_concat_2])

    print(df)

    df['unique_chars'] = df['enc_text'].apply(lambda x: len(set(x)))
    df['plain_text_length'] = df['plain_text'].apply(len)

    df_key1 = df[df['key'].apply(len) == 1]
    df_key2 = df[df['key'].apply(len) == 2]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plot1 = sns.scatterplot(data=df_key1, x='plain_text_length', y='unique_chars')
    plot1.set_xticks(range(df['plain_text_length'].min(), df['plain_text_length'].max() + 1))
    plot1.set_yticks(range(df['unique_chars'].min(), df['unique_chars'].max() + 1))
    plt.title('Chave de 1 caractere')
    plt.xlabel('Tamanho do texto de entrada')
    plt.ylabel('Quantidade de caracteres únicos do texto encriptado')

    plt.subplot(1, 2, 2)
    plot2 = sns.scatterplot(data=df_key2, x='plain_text_length', y='unique_chars')
    plot2.set_xticks(range(df['plain_text_length'].min(), df['plain_text_length'].max() + 1))
    plot2.set_yticks(range(df['unique_chars'].min(), df['unique_chars'].max() + 1))
    plt.title('Chave de 2 caracteres')
    plt.xlabel('Tamanho do texto de entrada')
    plt.ylabel('Quantidade de caracteres únicos do texto encriptado')

    plt.show()

    def letter_percentage(text):
        total_chars = len(text)
        counts = pd.Series(list(text)).value_counts(normalize=True) * 100
        return counts.reindex(sorted(counts.index), fill_value=0)

    df_percentages_list = [letter_percentage(enc_text) for enc_text in df['enc_text']]

    n = len(df_percentages_list)

    rows = int(np.ceil(n / 3))
    cols = 3

    plt.figure(figsize=(15, rows * 5))

    for i, percentages in enumerate(df_percentages_list):
        plt.subplot(rows, cols, i + 1)
        sns.barplot(x=percentages.index, y=percentages.values)
        plt.title(
            f'Distribuição de letras no texto encriptados para:\nTexto de entrada: {df["plain_text"].iloc[i]} e Chave: {df["key"].iloc[i]}')
        plt.xlabel('Letra')
        plt.ylabel('Porcentagem (%)')

    plt.tight_layout()
    plt.show()
