#!/bin/bash

# create directory
# mkdir full/
# mkdir full/metadata/
# mkdir full/pdf_parses/

wget -O LICENSE 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/LICENSE?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=vXSEkEjEuqKjx3f0fS36mUvbaNs%3D&Expires=1639425139'

wget -O RELEASE_NOTES 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/RELEASE_NOTES?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Kns0we%2BPJXSTTjLdhw4D82VL98s%3D&Expires=1639425139'

# wget -O full/metadata/metadata_0.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_0.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=HNU29WxNIDeL%2BMk0TG1G%2F2unZQY%3D&Expires=1639425139'

# wget -O full/metadata/metadata_1.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_1.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=2N2xOL3Fr4oNZujr%2BTtjGLuvt6E%3D&Expires=1639425139'

# wget -O full/metadata/metadata_10.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_10.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=jCYy1vDRZf9J01XpX9R5RMqWArc%3D&Expires=1639425139'

# wget -O full/metadata/metadata_11.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_11.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=6T%2F6PWmg%2Bd05BcZyMm4ei3fxvkg%3D&Expires=1639425139'

# wget -O full/metadata/metadata_12.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_12.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Jnji1Yqxku6xhPy2XEaVzpWOW6k%3D&Expires=1639425139'

# wget -O full/metadata/metadata_13.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_13.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=GQXJ6ld0yimizcKhYdo%2Bagb85vs%3D&Expires=1639425139'

# wget -O full/metadata/metadata_14.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_14.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=50SyPGhRHqaOiT9%2BvU21Eow7a04%3D&Expires=1639425140'

# wget -O full/metadata/metadata_15.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_15.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=QyimisWmys%2FNBFrJ3UgVmFu%2Bu3I%3D&Expires=1639425140'

# wget -O full/metadata/metadata_16.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_16.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=MBcm6HAkWL8asDlnvSPPeySsJic%3D&Expires=1639425140'

# wget -O full/metadata/metadata_17.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_17.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=fw8t72FL0tq2sGZyaTdOC8FZSEk%3D&Expires=1639425140'

# wget -O full/metadata/metadata_18.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_18.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=G1qdo42oThsRq0OzB8XIijU3fCI%3D&Expires=1639425140'

# wget -O full/metadata/metadata_19.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_19.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=TkA%2BbzjXMA8pXUqPBER%2Ffp4%2FyCY%3D&Expires=1639425140'

# wget -O full/metadata/metadata_2.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_2.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=n5LAJp5Zy2HljhccE7LfUIqMfT8%3D&Expires=1639425140'

# wget -O full/metadata/metadata_20.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_20.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=VeMWFPmWn62B3QhQOMJC%2Ff02jFU%3D&Expires=1639425140'

# wget -O full/metadata/metadata_21.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_21.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=a4WUvNQTGgawySgqKDECeol0LG0%3D&Expires=1639425140'

# wget -O full/metadata/metadata_22.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_22.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=dDcHP6k2Ba2Q1V%2F6iivrouZLmDI%3D&Expires=1639425140'

# wget -O full/metadata/metadata_23.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_23.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=PdkhZm5QHLi16EmqFHAzPqSX4LI%3D&Expires=1639425140'

# wget -O full/metadata/metadata_24.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_24.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=WutfYwqCk8VENmy3mvvWTww4WyM%3D&Expires=1639425140'

# wget -O full/metadata/metadata_25.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_25.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=L8rRbealnkf%2FOn5EKI899YGXjxk%3D&Expires=1639425141'

# wget -O full/metadata/metadata_26.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_26.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=arSY7drWnGYOp8ksIjn6wyYgxbk%3D&Expires=1639425141'

# wget -O full/metadata/metadata_27.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_27.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=YbpkQ3S1JCBeffqoLNCsmOPR7so%3D&Expires=1639425141'

# wget -O full/metadata/metadata_28.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_28.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=RRSfFnrAkI%2BkWRlTo0PflWAyqSM%3D&Expires=1639425141'

# wget -O full/metadata/metadata_29.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_29.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=pydB%2BeyoeutyS4iZJuoCud82BKU%3D&Expires=1639425141'

# wget -O full/metadata/metadata_3.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_3.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=P8UByTWUA1msQ33raI14Pwwx4I4%3D&Expires=1639425141'

# wget -O full/metadata/metadata_30.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_30.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=mYpwk5yS98MPfwprL0rBrAEhQ%2Bg%3D&Expires=1639425141'

# wget -O full/metadata/metadata_31.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_31.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=MlXlaLgixBz4TUhoiid3rCS1Zhg%3D&Expires=1639425141'

wget -O full/metadata/metadata_32.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_32.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=6Ses2KzL9h9bw%2FtY8nILeI%2BXdPg%3D&Expires=1639425141'

wget -O full/metadata/metadata_33.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_33.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=2TG1dQNePJBeOJ37fuDGaUQUPI0%3D&Expires=1639425141'

wget -O full/metadata/metadata_34.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_34.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=sE0JtW5JBiiFHTK4CtnoG%2FzzneQ%3D&Expires=1639425141'

wget -O full/metadata/metadata_35.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_35.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=7RFUlQU1GkIcTizkJuCYKsCx%2BU0%3D&Expires=1639425142'

wget -O full/metadata/metadata_36.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_36.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=KlCY6BzWTy14fZxHyV7y8%2FTM650%3D&Expires=1639425142'

wget -O full/metadata/metadata_37.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_37.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=xTUuFbgNNZ8WED%2F%2FlUu%2FcPYqr%2BE%3D&Expires=1639425142'

wget -O full/metadata/metadata_38.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_38.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=1lhugEr10zbohecaLOD%2FqT05c1M%3D&Expires=1639425142'

wget -O full/metadata/metadata_39.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_39.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=hvPGSgphRUzdHh94l1%2BXra7qRvE%3D&Expires=1639425142'

wget -O full/metadata/metadata_4.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_4.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=sVU6CGrlomoAY1lGLGgBR9gKjV0%3D&Expires=1639425142'

wget -O full/metadata/metadata_40.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_40.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=eo2FdYaNIBdAayMpskYMgmkDsuM%3D&Expires=1639425142'

wget -O full/metadata/metadata_41.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_41.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=cbOw8CpTWTsrRYxiZu9g%2FN0PgZk%3D&Expires=1639425142'

wget -O full/metadata/metadata_42.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_42.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=eCSygadlXvYnS2rnwmLk7bKDDvA%3D&Expires=1639425142'

wget -O full/metadata/metadata_43.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_43.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ik6QLOifLNgonSD5Hd3K1q3ZKAY%3D&Expires=1639425142'

wget -O full/metadata/metadata_44.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_44.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=I328BG2VWLJcljxr31iypuNLPPM%3D&Expires=1639425143'

wget -O full/metadata/metadata_45.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_45.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=u0cXIjmzYctO9QCALawEMhe7a5c%3D&Expires=1639425143'

wget -O full/metadata/metadata_46.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_46.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=C557YSk1WypAQC6SCMwyeSUY7ro%3D&Expires=1639425143'

wget -O full/metadata/metadata_47.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_47.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=BGqt5BnQN%2F1q61PLj4xodjbHJVk%3D&Expires=1639425143'

wget -O full/metadata/metadata_48.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_48.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=6CTVktOTG5%2Fsl3lgkjuLYHnfDCA%3D&Expires=1639425143'

wget -O full/metadata/metadata_49.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_49.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=12lCy6pWqBPduYSLc5dhW%2FpHlT4%3D&Expires=1639425143'

wget -O full/metadata/metadata_5.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_5.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=vEp2wE9kJTudxRGWedbi8yaloW4%3D&Expires=1639425143'

wget -O full/metadata/metadata_50.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_50.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=GvTQ1L%2BknWsWkhynsaXZJKUAW7o%3D&Expires=1639425143'

wget -O full/metadata/metadata_51.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_51.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=2DfEZ6ZFX5kI9ojs%2F%2BWriTcqNK4%3D&Expires=1639425143'

wget -O full/metadata/metadata_52.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_52.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ECimgmeqpyCWNDPUDTOpXzmBNTg%3D&Expires=1639425143'

wget -O full/metadata/metadata_53.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_53.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=30c59fnCIkdDX6ifPf0fAhexSY8%3D&Expires=1639425144'

wget -O full/metadata/metadata_54.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_54.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=prhFt%2FiSJOxwmemOS8pJhcSdvc0%3D&Expires=1639425144'

wget -O full/metadata/metadata_55.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_55.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=S1m%2Bnez4SJCJWcbq1gqUQXAjR2I%3D&Expires=1639425144'

wget -O full/metadata/metadata_56.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_56.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=xBGCPxeTnUdFfBNIaUyje%2BsPbUY%3D&Expires=1639425144'

wget -O full/metadata/metadata_57.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_57.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=yMC8E5BLaJnseolcZvMdJtx6P6M%3D&Expires=1639425144'

wget -O full/metadata/metadata_58.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_58.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=F%2BiWDuHAtfaKK7TUihELXkCMeS4%3D&Expires=1639425144'

wget -O full/metadata/metadata_59.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_59.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=4FJvEFbB%2FmsbtrLchjPFemcqoUg%3D&Expires=1639425144'

wget -O full/metadata/metadata_6.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_6.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ayrMuHpgyBVbsywdt8KM1ZPgJDY%3D&Expires=1639425144'

wget -O full/metadata/metadata_60.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_60.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=0oVNuhMHA7JAMqrVc56rtqCMGc4%3D&Expires=1639425144'

wget -O full/metadata/metadata_61.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_61.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Bcve6UR2iA4%2F719aoLtjlJ3d%2Bi8%3D&Expires=1639425144'

wget -O full/metadata/metadata_62.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_62.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=fQx3%2FwXjVULGa3vfSt7l5ipI6DU%3D&Expires=1639425145'

wget -O full/metadata/metadata_63.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_63.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=%2FkPKWzT18C4m4K6N8lx%2FwHDHd1c%3D&Expires=1639425145'

wget -O full/metadata/metadata_64.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_64.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=f5HOf6RkH5dSeZwfgyse31lmC7I%3D&Expires=1639425145'

wget -O full/metadata/metadata_65.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_65.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=yBak4CQlIiWa8HSkM0HITxCzp7M%3D&Expires=1639425145'

wget -O full/metadata/metadata_66.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_66.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ES5TLvg%2FcVvHiTt7%2FJPuMX0Xh24%3D&Expires=1639425145'

wget -O full/metadata/metadata_67.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_67.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Pfoyc2fSSqeAOHBKFumTrZzLIec%3D&Expires=1639425145'

wget -O full/metadata/metadata_68.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_68.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=yiMg5ppUDVUroNPW6RsGStoOOsY%3D&Expires=1639425145'

wget -O full/metadata/metadata_69.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_69.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=XLcpoGSlMoY12Jb0Wm1OQbzNOp4%3D&Expires=1639425145'

wget -O full/metadata/metadata_7.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_7.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=xdvUERn6V2N%2B3Bi6qdttbEIpVxs%3D&Expires=1639425145'

wget -O full/metadata/metadata_70.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_70.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=cy6C3KcV2zai9PNe7mkkvoAUfZ8%3D&Expires=1639425146'

wget -O full/metadata/metadata_71.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_71.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=OtU2Z9ec1kVo3kve0nYd0HNxzUw%3D&Expires=1639425146'

wget -O full/metadata/metadata_72.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_72.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ywaZFrOyC7KLZ7PYmZD4W05IurE%3D&Expires=1639425146'

wget -O full/metadata/metadata_73.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_73.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=o7s4wMLK4c8rW%2B07cCGR8oOmyfI%3D&Expires=1639425146'

wget -O full/metadata/metadata_74.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_74.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=kV3kl6M4YO%2BcI9HUgRw%2BJnbWvNI%3D&Expires=1639425146'

wget -O full/metadata/metadata_75.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_75.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=rJTPyskzl%2BfAQ46rq%2Byal0a64Po%3D&Expires=1639425146'

wget -O full/metadata/metadata_76.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_76.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=iqNHaQrnseT77BZNwfkJB22%2BPok%3D&Expires=1639425146'

wget -O full/metadata/metadata_77.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_77.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=6CYuzcUywdJmfu7L9UPOeyGouwc%3D&Expires=1639425146'

wget -O full/metadata/metadata_78.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_78.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=c%2B6c2M26lVL6tiNa1G3NG4OyZkw%3D&Expires=1639425147'

wget -O full/metadata/metadata_79.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_79.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Ds3Gl%2BiKevFLd7NaIe9loj40XwA%3D&Expires=1639425147'

wget -O full/metadata/metadata_8.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_8.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=xw7zx6g%2FF3%2BKBEr6fIibT%2BjW8As%3D&Expires=1639425147'

wget -O full/metadata/metadata_80.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_80.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=wEjBvLe83wMtLi9daQnk3vTrfGs%3D&Expires=1639425147'

wget -O full/metadata/metadata_81.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_81.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=njEMyJ23F6pa23MhE3rN1%2FdYzIc%3D&Expires=1639425147'

wget -O full/metadata/metadata_82.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_82.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Qbr%2BXMHtSnkVkStj%2BSsmlYQcvAk%3D&Expires=1639425147'

wget -O full/metadata/metadata_83.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_83.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=do%2Bye5kOsGyZOAGXWu39emvjm8A%3D&Expires=1639425147'

wget -O full/metadata/metadata_84.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_84.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=mbPNM4by1u1H10WbZ3MLVPLdZXY%3D&Expires=1639425147'

wget -O full/metadata/metadata_85.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_85.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=J5hYvZwDNrHESabUdIkLGge0B%2F0%3D&Expires=1639425147'

wget -O full/metadata/metadata_86.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_86.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=7LmCjWotktvyMYnEKPxtHKWYfPk%3D&Expires=1639425148'

wget -O full/metadata/metadata_87.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_87.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=TgFP26%2B1yNTzRo90gK89%2Bv8d7Uo%3D&Expires=1639425148'

wget -O full/metadata/metadata_88.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_88.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=MvRih83W8Dej5ICWe4foFsoJgpg%3D&Expires=1639425148'

wget -O full/metadata/metadata_89.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_89.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=rdjAqfsQdCHKFbtAh52%2F6zdpi2M%3D&Expires=1639425148'

wget -O full/metadata/metadata_9.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_9.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=KdUNqioI4ylmJLvBZ2azN%2B57shI%3D&Expires=1639425148'

wget -O full/metadata/metadata_90.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_90.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=rrByNTbptnoKCVfU3bh2buzaVxM%3D&Expires=1639425148'

wget -O full/metadata/metadata_91.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_91.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=VSyBvKKq6XT5QhRBrXfk7EOAerM%3D&Expires=1639425148'

wget -O full/metadata/metadata_92.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_92.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ptjxhmUpVE07Gh68uAKfBT0smZw%3D&Expires=1639425148'

wget -O full/metadata/metadata_93.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_93.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=2eFkpsjMr3zoHJ%2FDOZDUjYfkjiM%3D&Expires=1639425148'

wget -O full/metadata/metadata_94.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_94.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Kr90n0pc%2BFmo5ul6lGPO%2BqNeASg%3D&Expires=1639425149'

wget -O full/metadata/metadata_95.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_95.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Sgnx5tD7a2BRq6%2B5bVJNbUESk%2FU%3D&Expires=1639425149'

wget -O full/metadata/metadata_96.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_96.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=SwP%2Bqme1RhbhiAWhQwjrK3HfHYU%3D&Expires=1639425149'

wget -O full/metadata/metadata_97.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_97.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=KCYAPI9fy9m%2FQkCDGk9Enc3Ban8%3D&Expires=1639425149'

wget -O full/metadata/metadata_98.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_98.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=GdrzWs3uj4uNR0goaEZv0h68HzY%3D&Expires=1639425149'

wget -O full/metadata/metadata_99.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_99.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=30BH24vfLMfvBJtreRxyPJrAEGc%3D&Expires=1639425149'

# wget -O full/pdf_parses/pdf_parses_0.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_0.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=rD1u6QS9QPeA0DQX%2BJOYKnRm7S4%3D&Expires=1639425149'

# wget -O full/pdf_parses/pdf_parses_1.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_1.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=CgLr8pEcTXDchWTrLBy0xmTuu3w%3D&Expires=1639425149'

# wget -O full/pdf_parses/pdf_parses_10.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_10.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=rG2H3L5WhNhMkdYi%2Bh1G6%2FL88eI%3D&Expires=1639425150'

# wget -O full/pdf_parses/pdf_parses_11.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_11.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=5q6DqPeAR5m47oNvABNI4K5EmvI%3D&Expires=1639425150'

# wget -O full/pdf_parses/pdf_parses_12.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_12.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=vg0O49F2Bi%2FHyB79YT6TBraKDRA%3D&Expires=1639425150'

# wget -O full/pdf_parses/pdf_parses_13.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_13.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=yv3Ob5ToiYYqIRUcWFpPHyBe%2Bqc%3D&Expires=1639425150'

# wget -O full/pdf_parses/pdf_parses_14.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_14.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=9tS9unVFBHKjRiYhpK%2BfxeqWaZs%3D&Expires=1639425150'

# wget -O full/pdf_parses/pdf_parses_15.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_15.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=vaZpMFwZiDFyBSAItiVe7OD5AAI%3D&Expires=1639425150'

# wget -O full/pdf_parses/pdf_parses_16.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_16.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=6c7alHxVlGLTd0zWPKoXjDWQTzY%3D&Expires=1639425150'

# wget -O full/pdf_parses/pdf_parses_17.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_17.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=IUsrXq5YW19CvD6nYsjaCJDbn8k%3D&Expires=1639425150'

# wget -O full/pdf_parses/pdf_parses_18.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_18.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=nBuxhbz5p6p5CNmxqwFTRERxaaU%3D&Expires=1639425151'

# wget -O full/pdf_parses/pdf_parses_19.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_19.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=lt5FmxraTjygD943Lzt0uMiaxms%3D&Expires=1639425151'

# wget -O full/pdf_parses/pdf_parses_2.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_2.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=rED4z8%2BofieV889c37FTQ5rwp%2F0%3D&Expires=1639425151'

# wget -O full/pdf_parses/pdf_parses_20.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_20.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=aiEtfLRIM9l3PAN7GpIrm%2F5yHRQ%3D&Expires=1639425151'

# wget -O full/pdf_parses/pdf_parses_21.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_21.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=LFubBhIN5b64On7lEFOnjZpIl9o%3D&Expires=1639425151'

# wget -O full/pdf_parses/pdf_parses_22.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_22.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ssubDAlfsKK7YL2Xs76u3EyW10w%3D&Expires=1639425151'

# wget -O full/pdf_parses/pdf_parses_23.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_23.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=bCILVpjPrrm%2BnV4GUeNZDzZI1wo%3D&Expires=1639425151'

# wget -O full/pdf_parses/pdf_parses_24.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_24.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=0iv%2Fw6%2FjWJMeCLshlqq4yKTc4gQ%3D&Expires=1639425152'

# wget -O full/pdf_parses/pdf_parses_25.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_25.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ZBSCQBt6CNSa6ruk7VBb%2Fcg8ZYg%3D&Expires=1639425152'

# wget -O full/pdf_parses/pdf_parses_26.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_26.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=aRQQ4KDBV7yEOHLbjNkYw2LmrfI%3D&Expires=1639425152'

# wget -O full/pdf_parses/pdf_parses_27.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_27.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=p7Su7320tdSGbjFt%2Frbkhq8eu2k%3D&Expires=1639425152'

# wget -O full/pdf_parses/pdf_parses_28.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_28.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=wbCRG%2Fc3k2B%2BvrGTgPfU2fSOqBE%3D&Expires=1639425152'

# wget -O full/pdf_parses/pdf_parses_29.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_29.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=qwDRPF5pB%2BWSlqUg6OW6Tg3duoA%3D&Expires=1639425152'

# wget -O full/pdf_parses/pdf_parses_3.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_3.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=hNrCoW3bLWKR1f1A%2FEYlaCTzDkE%3D&Expires=1639425152'

# wget -O full/pdf_parses/pdf_parses_30.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_30.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=7qyFOP67qbLC%2FzR3WYgSSgiv3eE%3D&Expires=1639425153'

# wget -O full/pdf_parses/pdf_parses_31.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_31.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=chgxUPfkAZaPah2f9TCPftDS%2BhA%3D&Expires=1639425153'

# wget -O full/pdf_parses/pdf_parses_32.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_32.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=iVd7TWROsmQzhWnxgwoIF7BVOL4%3D&Expires=1639425153'

# wget -O full/pdf_parses/pdf_parses_33.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_33.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ExeulTVSGDYErspZV%2FevdlN2snk%3D&Expires=1639425153'

# wget -O full/pdf_parses/pdf_parses_34.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_34.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=H%2BLd4BSbC4m3%2B4%2B4ksaoUoW7IwQ%3D&Expires=1639425153'

# wget -O full/pdf_parses/pdf_parses_35.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_35.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=jDubSNpLXTBJ6I8dhBZWMZk2O6I%3D&Expires=1639425153'

# wget -O full/pdf_parses/pdf_parses_36.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_36.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=%2FFxW5kwN%2BX1vFCsFGFpl5sfVW40%3D&Expires=1639425154'

# wget -O full/pdf_parses/pdf_parses_37.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_37.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=3lni8OdcIUrX3kO%2F2%2BhnqivrSKw%3D&Expires=1639425154'

# wget -O full/pdf_parses/pdf_parses_38.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_38.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=WQP2Ku9gjc0BgG7nwipTsSMMe0M%3D&Expires=1639425154'

# wget -O full/pdf_parses/pdf_parses_39.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_39.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=vyGSLwUuoPhIdezcOfFkfqHYIhY%3D&Expires=1639425154'

# wget -O full/pdf_parses/pdf_parses_4.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_4.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=cUSnZZX5uunEqi521qMX%2FgN%2BiOY%3D&Expires=1639425154'

# wget -O full/pdf_parses/pdf_parses_40.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_40.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=cfYSpQR1ajVGPqgupU0o2CxCu4I%3D&Expires=1639425155'

# wget -O full/pdf_parses/pdf_parses_41.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_41.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=MF%2FzgeAg16igLAmX4uKcObpt69U%3D&Expires=1639425155'

# wget -O full/pdf_parses/pdf_parses_42.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_42.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=7PlB2Ga%2F8%2B26m8UQe4PD4v9P0j8%3D&Expires=1639425155'

# wget -O full/pdf_parses/pdf_parses_43.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_43.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=piqr3VX1H1PHWG5H541SfqyOkJQ%3D&Expires=1639425155'

# wget -O full/pdf_parses/pdf_parses_44.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_44.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Jxq4YCi6pU3NYUK%2BsgLOc3tbMaQ%3D&Expires=1639425155'

# wget -O full/pdf_parses/pdf_parses_45.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_45.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=v9uwStZqa2QigiHS%2BYy0kY32lfI%3D&Expires=1639425156'

# wget -O full/pdf_parses/pdf_parses_46.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_46.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=TsU7I0qfwt86DHt6%2FQhoNV4vcYc%3D&Expires=1639425156'

# wget -O full/pdf_parses/pdf_parses_47.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_47.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=2rTJ1rrW2bskaqPQ4btJNwa40u8%3D&Expires=1639425156'

# wget -O full/pdf_parses/pdf_parses_48.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_48.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=6f4%2F3f9eoDYPMg1v%2FdjpjBTovKc%3D&Expires=1639425156'

# wget -O full/pdf_parses/pdf_parses_49.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_49.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=bofaJINkf9xZePN%2FkoSIzsE1k5M%3D&Expires=1639425156'

# wget -O full/pdf_parses/pdf_parses_5.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_5.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=aiIbhHFO4qAhAAH59QmUOoenaDs%3D&Expires=1639425157'

# wget -O full/pdf_parses/pdf_parses_50.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_50.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=X%2FSBwUkoVifztpzkMZvpxTcF2ko%3D&Expires=1639425157'

# wget -O full/pdf_parses/pdf_parses_51.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_51.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=M%2F4RqqAB43Wk8IbrBnWcChMUDIU%3D&Expires=1639425157'

# wget -O full/pdf_parses/pdf_parses_52.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_52.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=HnthghKtMKtW9%2FsU51PDB6X4MEw%3D&Expires=1639425157'

# wget -O full/pdf_parses/pdf_parses_53.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_53.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=OpV1LTym3O4ifFsnJen3mlEXAEg%3D&Expires=1639425158'

# wget -O full/pdf_parses/pdf_parses_54.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_54.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=DtMllX37mfTzFDW%2B6rWklwY%2FGxM%3D&Expires=1639425158'

# wget -O full/pdf_parses/pdf_parses_55.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_55.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=MjVR9Ga1ZJVSuPWGgXfxfoWsNPU%3D&Expires=1639425158'

# wget -O full/pdf_parses/pdf_parses_56.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_56.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=05jgkwN8SXlRLrrmhItN%2B1ZLwRQ%3D&Expires=1639425159'

# wget -O full/pdf_parses/pdf_parses_57.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_57.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=k8vhe9VHG93H21tfGkSJDL0VLg4%3D&Expires=1639425159'

# wget -O full/pdf_parses/pdf_parses_58.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_58.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ogl5Hb6oCZRURcrhY2Cn5%2Bts1t4%3D&Expires=1639425159'

# wget -O full/pdf_parses/pdf_parses_59.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_59.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=WV6ZtzUj%2F%2BUWA4JHNpZDoMUU4Tc%3D&Expires=1639425160'

# wget -O full/pdf_parses/pdf_parses_6.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_6.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=QBFAJcpQyW54nrfn1oGM7NX8tvY%3D&Expires=1639425160'

# wget -O full/pdf_parses/pdf_parses_60.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_60.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=SrbPo8zfvXQNLg%2BYTBvJh%2B%2Fyz1w%3D&Expires=1639425161'

# wget -O full/pdf_parses/pdf_parses_61.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_61.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=npqarUbnX2eIhN31c1chrFCwFnw%3D&Expires=1639425161'

# wget -O full/pdf_parses/pdf_parses_62.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_62.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=PKUb6DjPD5nkFVC12fnDa0tdZ2s%3D&Expires=1639425162'

# wget -O full/pdf_parses/pdf_parses_63.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_63.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=EVepk3FKUdMJ5E9kNhi238uOnkw%3D&Expires=1639425162'

# wget -O full/pdf_parses/pdf_parses_64.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_64.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=bB%2FUV3eWf1VokGN56fOYSWP655s%3D&Expires=1639425162'

# wget -O full/pdf_parses/pdf_parses_65.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_65.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=jr8JH1AclFGNj1SYvV9E0xSirQo%3D&Expires=1639425163'

# wget -O full/pdf_parses/pdf_parses_66.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_66.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=dn171zzaW7qBAbZjf%2BzKA1aY9kg%3D&Expires=1639425163'

# wget -O full/pdf_parses/pdf_parses_67.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_67.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=fe913Ebo58hquJOGfWqkjh%2BkvtA%3D&Expires=1639425164'

# wget -O full/pdf_parses/pdf_parses_68.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_68.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=bdR1E1WaQncMMCcvMQtF4X7Oqn0%3D&Expires=1639425165'

# wget -O full/pdf_parses/pdf_parses_69.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_69.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=3r%2B7fgXl71S88vUGJz%2BdXlEWKr4%3D&Expires=1639425166'

# wget -O full/pdf_parses/pdf_parses_7.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_7.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=lmkDhKEQWZ94V%2FMGLWN019mc1xo%3D&Expires=1639425167'

# wget -O full/pdf_parses/pdf_parses_70.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_70.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=YcOGoQzAQVKgerdtjXEwd%2FA8JZk%3D&Expires=1639425168'

# wget -O full/pdf_parses/pdf_parses_71.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_71.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=%2B8%2FeckS55k45blkEKxLmM3QP%2FjE%3D&Expires=1639425169'

# wget -O full/pdf_parses/pdf_parses_72.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_72.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=gfY6ISn6DmYQfH0V1v1FVNp7zN4%3D&Expires=1639425170'

# wget -O full/pdf_parses/pdf_parses_73.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_73.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=wPWSXePoDSRSO5zQ%2BIQus1nfYDM%3D&Expires=1639425171'

# wget -O full/pdf_parses/pdf_parses_74.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_74.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=JUhjemCTwPaFshfCNsgtvKMYVnk%3D&Expires=1639425172'

# wget -O full/pdf_parses/pdf_parses_75.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_75.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=MMjiXa1WLq7xrLC9PN0JO33YqZQ%3D&Expires=1639425173'

# wget -O full/pdf_parses/pdf_parses_76.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_76.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=3MyoyvczHGyuShRMklk24A%2B91Pc%3D&Expires=1639425174'

# wget -O full/pdf_parses/pdf_parses_77.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_77.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=wR8%2BBsYUptkCFPimS41wb86vChU%3D&Expires=1639425174'

# wget -O full/pdf_parses/pdf_parses_78.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_78.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=q0raFK7XWshkB%2FTsGUXfXL2mNv4%3D&Expires=1639425175'

# wget -O full/pdf_parses/pdf_parses_79.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_79.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=dBPrOZRYxdci84fyztag%2BQOWfxw%3D&Expires=1639425176'

# wget -O full/pdf_parses/pdf_parses_8.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_8.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=qd2sUEXKc82H5iK4561O6n4YbW0%3D&Expires=1639425177'

# wget -O full/pdf_parses/pdf_parses_80.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_80.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=9zTfUkvMVwtUfkaYGd%2B1ykbJbAU%3D&Expires=1639425178'

# wget -O full/pdf_parses/pdf_parses_81.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_81.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=cRvsVrmmPjM7mEi8zXOSpA7vEUc%3D&Expires=1639425179'

# wget -O full/pdf_parses/pdf_parses_82.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_82.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=p%2BIIkO1cZ7j04460hcB2bw9h5gc%3D&Expires=1639425180'

# wget -O full/pdf_parses/pdf_parses_83.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_83.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=H8u6JNyxscYitFLpEtbFGi15Bpo%3D&Expires=1639425181'

# wget -O full/pdf_parses/pdf_parses_84.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_84.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=4UCE%2FI9lPhVZrRqogonNyG8ViDo%3D&Expires=1639425182'

# wget -O full/pdf_parses/pdf_parses_85.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_85.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=vFFksvPbcOshqtkW%2Bh1wrSdk9fY%3D&Expires=1639425183'

# wget -O full/pdf_parses/pdf_parses_86.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_86.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=XtsA%2FbX6Z7TDbOZZU6NWfFnCcLw%3D&Expires=1639425184'

# wget -O full/pdf_parses/pdf_parses_87.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_87.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ipBPMrCsyiClDD5HXtWMtmDdFXE%3D&Expires=1639425184'

# wget -O full/pdf_parses/pdf_parses_88.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_88.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=iwuAN1ltePIk0ZRYImi4XfyziAs%3D&Expires=1639425185'

# wget -O full/pdf_parses/pdf_parses_89.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_89.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=EkHJLvLrojLhNFzuQAJj1uaFgZY%3D&Expires=1639425186'

# wget -O full/pdf_parses/pdf_parses_9.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_9.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=XBxTMGJ47z4H3kww5Ox2ABSfdjc%3D&Expires=1639425187'

# wget -O full/pdf_parses/pdf_parses_90.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_90.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=E0bRHtm2Th1beWqiv5EMZiGmvpQ%3D&Expires=1639425187'

# wget -O full/pdf_parses/pdf_parses_91.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_91.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=SWFc4faToxvjYRMYr2sC8gSKEdk%3D&Expires=1639425188'

# wget -O full/pdf_parses/pdf_parses_92.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_92.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=rpsgFdB5QNTXj8m9HoCC3O6noBk%3D&Expires=1639425189'

# wget -O full/pdf_parses/pdf_parses_93.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_93.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=EJ2W1htATIf8bHZOj3QTo%2FGInc0%3D&Expires=1639425190'

# wget -O full/pdf_parses/pdf_parses_94.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_94.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=6ctLpwz6haiors32H5eYezeaXEw%3D&Expires=1639425191'

# wget -O full/pdf_parses/pdf_parses_95.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_95.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=FWZSzA9WlPuJUspQWOws5fh05So%3D&Expires=1639425192'

# wget -O full/pdf_parses/pdf_parses_96.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_96.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=jkvTJShgiScjlTeuE9CtcLbwlyc%3D&Expires=1639425193'

# wget -O full/pdf_parses/pdf_parses_97.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_97.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=GLMOqAii%2BwXgM6kI0WezN%2BAfY%2Bc%3D&Expires=1639425193'

# wget -O full/pdf_parses/pdf_parses_98.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_98.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=4QnM%2F80OfXOdhlRtavDjyz7gePE%3D&Expires=1639425194'

# wget -O full/pdf_parses/pdf_parses_99.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_99.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=eoJ9JlNbepxMx3te0xoxYGVbqRw%3D&Expires=1639425195'

