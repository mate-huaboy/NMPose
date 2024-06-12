from collections import OrderedDict
import pickle
import numpy as np
import pandas as pd
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"文件 '{file_path}' 未找到.")
        return None
    except Exception as e:
        print(f"读取文件 '{file_path}' 时出现错误: {e}")
        return None

# 示例用法
file_path = './verify_idea/a6-cPnP-lm13_lm_13_test_errors.pkl'
# file_path1='output39/gdrn/lm/a6_cPnP_lm13/inference_model_final/lm_13_test/a6-cPnP-lm13_lm_13_test_recalls.pkl'
# file_path2='output39/gdrn/lm/a6_cPnP_lm13/inference_model_final/lm_13_test/a6-cPnP-lm13_lm_13_test_preds.pkl'
loaded_data = load_pickle(file_path)
# loaded_data1 = load_pickle(file_path1)
# loaded_data2 = load_pickle(file_path2)

##旋转上的数据分析
if loaded_data is not None:
    print(f"成功读取.pkl文件 '{file_path}' 的内容:")
    #处理数据
    obj_names = sorted(list(loaded_data.keys()))
    #需要处理re的数据，阈值从0-10，20个数据
    precisions = OrderedDict()
    arr=np.arange(0,10,0.5)
    for obj_name in obj_names:
        precisions[obj_name]={a:[] for a in arr}
        for a in arr:
            precisions[obj_name][a].extend(loaded_data[obj_name]['re']<a)
    #求平均
    resu=[]
    for a in arr:
        line=[]
        for obj_name in obj_names:
            res=precisions[obj_name][a]
            line.append(np.mean(res))
        resu.append(np.mean(line))
    #保存到excel========
    df = pd.DataFrame({'Threshold': arr, 'Ours+Net': resu})
    excel_output_path = './file_Net.xlsx'
    df.to_excel(excel_output_path, index=False)
    #===========改成以追加的方式写入
    # existing_excel_file='./file.xlsx'
    # df=pd.read_excel(existing_excel_file)
    # df_append=pd.DataFrame({"Ours+Net":resu})
    # df=pd.concat([df,df_append],axis=1)
    # with pd.ExcelWriter(existing_excel_file, engine='openpyxl', mode='a',if_sheet_exists='replace') as writer:
    #     df.to_excel(writer, index=False, startrow=0, header=False)
