#!/bin/bash
git add /home/wangye/YeProject/code/*
git commit -m $1
git push origin main

echo "代码上传完成"
