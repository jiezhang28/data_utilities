import pandas as pd

def list_stack(frame, column, stack_col=None, split_char=',', drop_dup=False):
	"""
	Transform column where values are char-separated list into column where each list element has its own row

	frame : pandas dataframe
	column : name of dataframe column containing char-separated list
	stack_col : new name of transformed column. DEFAULT: old_name + '_stack'
	split_char : specific character used to separate values. DEFAULT: comma
	drop_dup : drop any duplicate rows after stacking. DEFAULT: False
	"""

	stack_col = stack_col if stack_col else column + '_stack' 

	df_st = pd.DataFrame(frame[column].str.split(split_char).tolist(), index=frame.index)
	df_st = df_st.stack() \
				.reset_index() \
				.drop('level_1', axis=1) \
				.rename(columns={0:stack_col})

	df_st[stack_col] = df_st[stack_col].str.strip()

	df_st = pd.merge(df_st, frame, left_on='level_0', right_index=True) \
				.drop('level_0', axis=1)

	if drop_dup: df_st = df_st.drop_duplicates()

	return df_st