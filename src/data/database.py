import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

Base = declarative_base()

class Board(Base):
    __tablename__ = 'boards'
    id = Column(Integer, primary_key=True)
    week_number = Column(Integer, unique=True)
    upload_date = Column(DateTime, default=datetime.datetime.now)
    image_path = Column(String)

class SquareValue(Base):
    __tablename__ = 'square_values'
    id = Column(Integer, primary_key=True)
    board_id = Column(Integer, ForeignKey('boards.id'))
    grid_index = Column(Integer)
    reward_amount = Column(Float, nullable=True) 

# Setup Engine
db_dir = os.path.join(os.getcwd(), 'data')
os.makedirs(db_dir, exist_ok=True)
db_path = os.path.join(db_dir, "database.db")
engine = create_engine(f'sqlite:///{db_path}')

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def save_results(reward_list, img_path, week_num):
    session = Session()
    try:
        existing_board = session.query(Board).filter_by(week_number=week_num).first()
        if existing_board:
            print(f"Week {week_num} already exists in DB. Skipping insert.")
            return existing_board.id

        new_board = Board(image_path=img_path, week_number=week_num)
        session.add(new_board)
        session.flush() 

        for i, value in enumerate(reward_list):
            safe_value = float(value) if pd.notnull(value) else None
            square = SquareValue(board_id=new_board.id, grid_index=i, reward_amount=safe_value)
            session.add(square)
            
        session.commit()
        print(f"Successfully saved Week {week_num} to database.")
        return new_board.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_history_df():
    query = """
    SELECT 
        b.week_number,
        s.grid_index,
        s.reward_amount as reward
    FROM square_values s
    JOIN boards b ON s.board_id = b.id
    ORDER BY b.week_number ASC, s.grid_index ASC
    """
    return pd.read_sql(query, con=engine)