from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
import pandas as pd

Base = declarative_base()

class Board(Base):
    __tablename__ = 'boards'
    id = Column(Integer, primary_key=True)
    upload_date = Column(DateTime, default=datetime.datetime.now)
    image_path = Column(String)

class SquareValue(Base):
    __tablename__ = 'square_values'
    id = Column(Integer, primary_key=True)
    board_id = Column(Integer, ForeignKey('boards.id'))
    grid_index = Column(Integer)
    reward_amount = Column(Integer)


engine = create_engine('sqlite:///../data/database/rewards.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def save_results(reward_list, img_path):
    session = Session()
    try:
        new_board = Board(image_path=img_path)
        session.add(new_board)
        session.flush()

        for i, value in enumerate(reward_list):
            square = SquareValue(board_id=new_board.id, grid_index=i, reward_amount = value)
            session.add(square)
        session.commit()
        return new_board.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_history_df():
    return pd.read_sql('SELECT * FROM square_values', con=engine)