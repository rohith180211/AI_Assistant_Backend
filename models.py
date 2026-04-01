from sqlalchemy import Column, String
from database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    path = Column(String)
    status = Column(String)