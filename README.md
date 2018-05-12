# Multi Candidate Thai Word Segmentation

Most existing word segmentation methods output
one single segmentation solution. This project provides an analysis
of word segmentation performance when more than one solutions are taken into account

## Prerequisites
### Weights & Dictionaries

Weights & Dictionaries can be download from this [link](https://goo.gl/hsgn8q)

### Required libraries
  - Python 3
  - pytorch 0.3.1
  - numpy
  - pickle

### Corpus

Text corpus from [InterBEST 2009/2010](https://www.nectec.or.th/corpus/index.php?league=pm)



### Usage


```
txt = 'เมื่อหนุ่มคนดังกล่าวขับ'
one_candidate(txt)
multi_candidate(txt)
```
Output of one candiate
```
เมื่อ|หนุ่ม|คน|ดัง|กล่าว|ขับ|
```
Output of multi-candidate
```
เมื่อหนุ่มคน|ดังกล่าวขับ
เมื่อหนุ่มคน|ดังกล่าว|ขับ
เมื่อหนุ่มคน|ดัง|กล่าว|ขับ
เมื่อหนุ่ม|คน|ดัง|กล่าว|ขับ
เมื่อ|หนุ่ม|คน|ดัง|กล่าว|ขับ
เมื่อ|หนุ่ม|คน|ดัง|กล่าว|ขับ|
เมื่อ|ห|นุ่ม|คน|ดัง|กล่าว|ขับ|
เมื่อ|ห|นุ่ม|คน|ดัง|ก|ล่าว|ขับ|
เมื่อ|ห|นุ่ม|คน|ดัง|ก|ล่า|ว|ขับ|
เมื่อ|ห|นุ่|ม|คน|ดัง|ก|ล่า|ว|ขับ|
เมื่อ|ห|นุ่|ม|ค|น|ดัง|ก|ล่า|ว|ขับ|
```








#

## Author

* **Theerapt Lapjaturapit**


