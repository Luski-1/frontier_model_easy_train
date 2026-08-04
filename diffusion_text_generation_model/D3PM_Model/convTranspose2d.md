# F.conv_transpose2d有两种理解方式
- 
    1. 对x（2*2）的每个元素，单独逐元素相乘kernal（假设3*3），那么每个元素变成3*3大小
    2. 随后根据stride进行排布，例如处于[0,0]的元素要霸占[0-2, 0-2]位置，处于[0,1]的的元素要霸占[0-2,1-3]位置，隔了stride步。
    3. 同理[1,0]的元素要霸占[1-3,0-2]位置，[1,1]的元素要霸占[1-3,1-3]位置
    4. 重叠位置的数值进行相加即可
- 
    1. 在每个元素之间插入stride-1个0
    2. 旋转卷积核180°
    3. 两边各做kernal_size - 1 - padding的填充0向量，随后进行stride固定=1的普通卷积
        PS：padding在F.conv_transpose2d的参数中是代表要裁剪两边各多少元素，output_padding是仅在最后结果的右边/下边填充多少0元素
    4. 还要理解卷积和转置卷积的分辨率计算公式
    5. 卷积 = (input_size + 2 * padding - kernal_size) / 2
    6. 转置卷积 = (input_size - 1) * stride - 2 * padding + kernal_size  + output_padding