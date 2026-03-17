import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

X_df = pd.read_excel('data/WIO_table(2020).xlsx', sheet_name="X", index_col=0)
WI_df = pd.read_excel('data/WIO_table(2020).xlsx', sheet_name="WⅠ", index_col=[0,1] ,header=1)
WII_df = pd.read_excel('data/WIO_table(2020).xlsx', sheet_name="WⅡ", index_col=[0,1] )
y_df = pd.read_excel('data/WIO_table(2020).xlsx', sheet_name="y")
Wf_df = pd.read_excel('data/WIO_table(2020).xlsx', sheet_name="Wf", index_col=[0,1] )
Sfinal_df = pd.read_excel('data/WIO_table(2020).xlsx', sheet_name="S")
S = Sfinal_df.iloc[1:5, 1:28].astype(float).values  # S: (4 x 28)
waste_columns = ['Landfill', 'Incineration', 'Anaerobic digestion', 'Composting']
x_I = X_df.loc["TI", X_df.columns[:42]].astype(float).values    # Total output vector (length 42)
sw = np.dot(S, WI_df)
swf = np.dot(S, Wf_df)

# Calculate domestic product ratio
io_data = pd.read_excel("data/WIO_table(2020).xlsx", sheet_name="io_data", index_col=0)
io_domestic = pd.read_excel("data/WIO_table(2020).xlsx", sheet_name="io_domestic", index_col=0)
io_data.columns = io_data.columns.astype(str).str.strip()
common_cols = io_data.columns.intersection(io_domestic.columns)
common_idx = io_data.index.intersection(io_domestic.index)
d = io_domestic.loc[common_idx, common_cols] / io_data.loc[common_idx, common_cols]
d = d.fillna(0).replace([float('inf'), -float('inf')], 0)
d_a = d.loc[:,[str(i) for i in range(1,43)]]
d_TC = d.loc[:,["TC"]]
d_FU = d.loc[:,["FU201","FU202"]]

#Final demand
Y_I = y_df.select_dtypes(include=[np.number])
Y_I.iloc[:, :3] = Y_I.iloc[:, :3].values * d_TC.values
Y_I.iloc[:, 3:5] = Y_I.iloc[:, 3:5].values * d_FU.values   # Share adjustment
y_I =Y_I.sum(axis=1).values
diag_y_I = np.diag(y_I)

# Waste final consumption emissions
Y_II= Wf_df.select_dtypes(include=[np.number])
y_II = Y_II.sum(axis=1).values
diag_y_II = np.diag(y_II)

# A_I (Industry to industry)
Z_AI = X_df.iloc[:42,:42].astype(float).values  # Input matrix (42x42)
# x_I = np.array(x_I)*np.array(d)
Z_AI_d = d_a * Z_AI
xw = X_df[waste_columns].iloc[:42].astype(float).values
x_I = Z_AI_d.sum(axis=1) + xw.sum(axis=1) + Y_I.values.sum(axis=1) 
x_I = np.array(x_I)
A_I = Z_AI_d / x_I [np.newaxis, :]  # Column-wise division

# A_II (Industry to waste treatment)
Z_AII = X_df[waste_columns].iloc[:42].astype(float).values  # 42x4
x_II= ((S @ np.array(WI_df)).sum(axis=1) + (S @ np.array(WII_df)).sum(axis=1) + (S @ np.array(Wf_df)).sum(axis=1)).T
#x_II = X_df[waste_columns].iloc[48].astype(float).values   
A_II = Z_AII / x_II [np.newaxis, :]

# G_I (Waste emission per unit output)
G_I = WI_df / x_I[np.newaxis, :]

# G_II (Waste treatment per unit treatment output)
G_II = WII_df / x_II[np.newaxis, :]

SG_I = np.dot(S, G_I)     # (4 x 42)
SG_II = np.dot(S, G_II)   # (4 x 4)

# Construct simultaneous matrix M = [[I - A_I, -A_II], [-SG_I, I - SG_II]]
I_AI = np.eye(A_I.shape[0]) - A_I    # (42 x 42)
I_SGII = np.eye(SG_II.shape[0]) - SG_II # (4 x 4)
top_block = np.hstack([I_AI, -A_II])    # (42 x 46)
bottom_block = np.hstack([-SG_I, I_SGII])  # (4 x 46)
M = np.vstack([top_block, bottom_block])  # (46 x 46)
M_inv = np.linalg.inv(M)  # Inversion

# Decompose into 4 sub-blocks
B_I_I   = M_inv[:42, :42]   # Top-left block (42 x 42)
B_I_II  = M_inv[:42, 42:]   # Top-right block (42 x 4)
B_II_I  = M_inv[42:, :42]   # Bottom-left block (4 x 42)
B_II_II = M_inv[42:, 42:]   # Bottom-right block (4 x 4)
    
# Waste footprint (by treatment type)
# x_II = B_II,I * y_I + B_II,II * S * y_II
x_II_total = np.dot(B_II_I, y_I) + np.dot(np.dot(B_II_II,S,), y_II)
x_II_p = np.dot(B_II_I, Y_I) + np.dot(np.dot(B_II_II,S,), Y_II)
x_II_n= np.dot(B_II_I, diag_y_I)
x_II_m=np.dot(np.dot(B_II_II,S), diag_y_II)

# Waste footprint (by waste type)
# w = (G_I * B_I,I + G_II * B_II,I) * y_I + (G_I * B_I,II + G_II * B_II,II) * S * y_II + y_II
C_I = np.dot(G_I, B_I_I) + np.dot(G_II, B_II_I)
C_II = np.dot(G_I, B_I_II) + np.dot(G_II, B_II_II)
w_total = np.dot(C_I, y_I) + np.dot(np.dot(C_II, S) + np.eye(27, 27), y_II)
#w_total = np.dot(np.dot(G_I, B_I_I), y_I) + np.dot(np.dot(G_II, B_II_I), y_I) + np.dot(np.dot(np.dot(G_I, B_I_II), S),y_II) + np.dot(np.dot(np.dot(G_II, B_II_II), S),y_II) + y_II
w_p = np.dot(C_I, Y_I) + np.dot(np.dot(C_II, S) + np.eye(27, 27), Y_II)
#w_p = np.dot(np.dot(G_I, B_I_I), Y_I) + np.dot(np.dot(G_II, B_II_I), Y_I) + np.dot(np.dot(np.dot(G_I, B_I_II), S),Y_II) + np.dot(np.dot(np.dot(G_II, B_II_II), S),Y_II) + Y_II    
w_n = np.dot(C_I, diag_y_I) 
#w_n = np.dot(np.dot(G_I, B_I_I), diag_y_I) + np.dot(np.dot(G_II, B_II_I), diag_y_I)     
w_m = np.dot(np.dot(C_II, S) + np.eye(27, 27), diag_y_II)
#w_m = np.dot(np.dot(np.dot(G_I, B_I_II), S),diag_y_II) + np.dot(np.dot(np.dot(G_II, B_II_II), S),diag_y_II) + diag_y_II           

# Verify total output and footprint
w_total1 = WI_df.sum().sum() + WII_df.sum().sum() + Wf_df.sum().sum()
w_total2 = w_p.sum().sum()   # w_total.sum(axis=0)
print(f"Total waste verification: Original={w_total1} 10k tons, Footprint={w_total2} 10k tons, Difference={w_total2-w_total1} 10k tons")

# Waste footprint (by treatment type) 
w_ldf = np.dot((np.diag(Sfinal_df.iloc[1, 1:28])) , w_n)     #Landfill
w_inc = np.dot((np.diag(Sfinal_df.iloc[2, 1:28])) , w_n)     #Incineration
w_com = np.dot((np.diag(Sfinal_df.iloc[3, 1:28])) , w_n)     #Composting
w_AD = np.dot((np.diag(Sfinal_df.iloc[4, 1:28])) , w_n)      #Anaerobic digestion

WI_df1 = pd.read_excel('data/WIO_table(2020).xlsx', sheet_name="WⅠ" )
Wf_df1 = pd.read_excel('data/WIO_table(2020).xlsx', sheet_name="Wf" )
n_labels = WI_df1.columns[2:]  # 42 industry sector names   
m_labels = WI_df1.iloc[1:, 1].tolist()   # 27 waste category names
p_labels = Wf_df1.columns[2:]  # p final demand sector names

output_excel = "result/result_net_p.xlsx"  # Demand-driven results
with pd.ExcelWriter(output_excel) as writer:
    for i, dname in enumerate(p_labels):
       # === xII_n: Industry sector × Treatment type footprint driven by current demand sector
       y = Y_I.iloc[:, i].values
       diag_y = np.diag(y)
       x_ii_n = np.dot(B_II_I, diag_y).T  # (42 x 4)
       df_xii_n = pd.DataFrame(x_ii_n, index=n_labels, columns=waste_columns)
       df_xii_n.to_excel(writer, sheet_name=f"xII_n_{dname}")
       
    for i, dname in enumerate(p_labels):
        # === w_n: Waste type × Industry sector footprint driven by current demand sector
        y = Y_I.iloc[:, i].values
        diag_y = np.diag(y)
        wn = np.dot(C_I, diag_y)  # (27 x 42)
        df_wn = pd.DataFrame(wn, index=m_labels, columns=n_labels)
        df_wn.to_excel(writer, sheet_name=f"w_n_{dname}")

    for i, dname in enumerate(p_labels):
        # === xII_m: Treatment type × Waste type footprint driven by current demand sector (post-consumption waste)
        if i < 2:
            y_ii = Y_II.iloc[:, i].values
            diag_y_ii = np.diag(y_ii)
            x_ii_m = np.dot(np.dot(B_II_II, S), diag_y_ii)  # (4 x 27)
            df_xii_m = pd.DataFrame(x_ii_m, index=waste_columns, columns=m_labels)
            df_xii_m.to_excel(writer, sheet_name=f"xII_m_{dname}")
           
# === Sum column totals of xII_m (Treatment type × Waste type) for first two demand sectors → Total post-consumption waste footprint (27×2)
w_m_sum = pd.DataFrame(index=m_labels)
for i, dname in enumerate(p_labels[:2]):  # Only first two
    y_ii = Y_II.iloc[:, i].values
    diag_y_ii = np.diag(y_ii)
    x_ii_m = np.dot(np.dot(B_II_II, S), diag_y_ii)  # (4 x 27)
    df_xii_m = pd.DataFrame(x_ii_m, index=waste_columns, columns=m_labels)
    w_m_sum[dname] = df_xii_m.sum(axis=0)  # Sum over treatment types (column-wise)

w_m_ldf = np.dot((np.diag(Sfinal_df.iloc[1, 1:28])) , w_m_sum)     #Landfill
w_m_inc = np.dot((np.diag(Sfinal_df.iloc[2, 1:28])) , w_m_sum)     #Incineration
w_m_com = np.dot((np.diag(Sfinal_df.iloc[3, 1:28])) , w_m_sum)     #Composting
w_m_AD = np.dot((np.diag(Sfinal_df.iloc[4, 1:28])) , w_m_sum)      #Anaerobic digestion 

with pd.ExcelWriter("result/result_net_all.xlsx") as writer:
    pd.DataFrame(x_II_total, index=waste_columns, columns=['xII_total']).to_excel(writer, sheet_name="xII_total")
    pd.DataFrame(x_II_p, index=waste_columns, columns=p_labels).to_excel(writer, sheet_name="xII_p")
    pd.DataFrame(x_II_n, index=waste_columns, columns=n_labels).to_excel(writer, sheet_name="xII_n")
    pd.DataFrame(x_II_m, index=waste_columns, columns=m_labels).to_excel(writer, sheet_name="xII_m")
    pd.DataFrame(w_total, index=m_labels, columns=['w_total']).to_excel(writer, sheet_name="w_total")
    pd.DataFrame(w_p, index=m_labels, columns=p_labels).to_excel(writer, sheet_name="w_p")
    pd.DataFrame(w_n, index=m_labels, columns=n_labels).to_excel(writer, sheet_name="w_n")
    pd.DataFrame(w_m_sum).to_excel(writer, sheet_name="w_m_consumption")
    pd.DataFrame(w_m, index=m_labels, columns=m_labels).to_excel(writer, sheet_name="w_m")
    pd.DataFrame(w_ldf, index=m_labels, columns=n_labels).to_excel(writer, sheet_name="ldf")
    pd.DataFrame(w_m_ldf, index=m_labels , columns=p_labels[:2]).to_excel(writer, sheet_name="ldf_consumption")
    pd.DataFrame(w_inc, index=m_labels, columns=n_labels).to_excel(writer, sheet_name="inc")
    pd.DataFrame(w_m_inc, index=m_labels , columns=p_labels[:2]).to_excel(writer, sheet_name="inc_consumption")
    pd.DataFrame(w_com, index=m_labels, columns=n_labels).to_excel(writer, sheet_name="AD")
    pd.DataFrame(w_m_com, index=m_labels , columns=p_labels[:2]).to_excel(writer, sheet_name="AD_consumption")
    pd.DataFrame(w_AD, index=m_labels, columns=n_labels).to_excel(writer, sheet_name="com") 
    pd.DataFrame(w_m_AD, index=m_labels , columns=p_labels[:2]).to_excel(writer, sheet_name="com_consumption")
    pd.DataFrame(sw, index=waste_columns, columns=n_labels).to_excel(writer, sheet_name="sw1")
    pd.DataFrame(swf, index=waste_columns, columns=p_labels).to_excel(writer, sheet_name="swf")


