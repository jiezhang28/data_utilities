"""
Various tools and convenience functions for common data tasks
"""
import numpy as np

def df_tosql_upsert(df, table_name, db_engine, chunk_size=None):
	"""
	DataFrame.to_sql function in pandas does not support upsert operation.
	This function will perform an upsert on a MySQL table only.
	db_engine must use the mysqlconnector SQLAlchemy dialect.
	Dataframe column names must match with that of the database table.

	Parameters
	----------
	df : Pandas dataframe
	table_name : Name of MySQL table
	db_engine : SQLAlchemy engine created using the MySQL dialect
	chunk_size : Rows to be loaded in a single insert. By default, all rows will be loaded.
	"""

	# TODO: make this campatible with postgres dialect
	
	from sqlalchemy import MetaData, Table
	from sqlalchemy.dialects.mysql import insert
	from sqlalchemy.dialects.mysql.mysqlconnector import MySQLDialect_mysqlconnector
	from sqlalchemy.dialects.mysql.pymysql import MySQLDialect_pymysql

	supported_dialects = (MySQLDialect_mysqlconnector, MySQLDialect_pymysql)
	assert(isinstance(db_engine.dialect, supported_dialects)), 'SQLAlchemy dialect is unsupport: {}'.format(type(db_engine.dialect))

	metadata = MetaData()
	table = Table(table_name, metadata, autoload=True, autoload_with=db_engine)
	table_columns = [col.name for col in table.columns]
	df_columns = df.columns

	assert(len(df_columns)<=len(table_columns)), 'Number of columns in dataframe is greater than number of columns in table: {} vs {}'.format(len(df_columns), len(table_columns))

	for col in df_columns: 
		assert(col in table_columns), 'Column in dataframe not found in columns in table: {}'.format(col)

	rows = len(df)
	if rows==0: return

	if chunk_size==None:
		chunk_size = rows
	else:
		assert(chunk_size>0)

	chunks = int(rows/chunk_size) + 1

	data = df_to_dict(df)
	with db_engine.begin() as conn:
		for i in range(chunks):
			start = i * chunk_size
			end = min((i+1) * chunk_size, rows)
			if start >= end: break

			chunk_data = data[start:end]
			ins = insert(table).values(chunk_data)
			ins_upsert = ins.on_duplicate_key_update(**dict(zip(df_columns, [ins.inserted[c] for c in df_columns])))
			conn.execute(ins_upsert)

def df_to_dict(df):
	"""
	Copied directly from pandas.io.sql.SQLTable.insert_data with slight modifications.
	Converts data in dataframe into 'records' dictionary list while converting pandas 
	Timestamp data type to python datatime datatype for better compabilities with SQL operations.
	Note: DataFrame.to_dict(orient='records') does not convert the Timestamp data type.	

	Parameters:
	--------
	df : Pandas dataframe

	"""
	from pandas.compat import text_type
	from pandas.core.dtypes.missing import isna

	column_names = list(map(text_type, df.columns))
	ncols = len(column_names)
	data_list = [None] * ncols
	blocks = df._data.blocks

	for b in blocks:
		if b.is_datetime:
			# return datetime.datetime objects
			if b.is_datetimetz:
				# GH 9086: Ensure we return datetimes with timezone info
				# Need to return 2-D data; DatetimeIndex is 1D
				d = b.values.to_pydatetime()
				d = np.expand_dims(d, axis=0)
			else:
				# convert to microsecond resolution for datetime.datetime
				d = b.values.astype('M8[us]').astype(object)
		else:
			d = np.array(b.get_values(), dtype=object)

		# replace NaN with None
		if b._can_hold_na:
			mask = isna(d)
			d[mask] = None

		for col_loc, col in zip(b.mgr_locs, d):
			data_list[col_loc] = col

	#return column_names, data_list
	return [dict(zip(column_names, row)) for row in zip(*data_list)]