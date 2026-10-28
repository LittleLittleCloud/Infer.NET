#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace lmhead_affine8 {
constexpr uint32_t Columns=2048, Rows=248320, Tile=4;
inline bool geometry(uint32_t experts,uint32_t rows,uint32_t columns,uint32_t bits,uint32_t group) {
    return experts==1&&rows==Rows&&columns==Columns&&bits==8&&group==64;
}
struct Layout {
    uint32_t rows;
    size_t paddedRows,codesBytes,scalesOffset,biasesOffset,bytes;
    explicit Layout(uint32_t n):rows(n),paddedRows((size_t(n)+3)/4*4),
        codesBytes(paddedRows*Columns),scalesOffset(codesBytes),
        biasesOffset(scalesOffset+paddedRows*32*2),bytes(biasesOffset+paddedRows*32*2) {}
    size_t code(uint32_t row,uint32_t column) const {
        return (((size_t(row/4)*8+column/256)*32+(column%256)/8)*4+row%4)*8+column%8;
    }
    size_t parameter(uint32_t row,uint32_t group) const {
        return (size_t(row/4)*32+group)*4+row%4;
    }
};
inline bool pack(uint32_t rows,const void *codes,size_t codesBytes,const void *scales,size_t scalesBytes,
    const void *biases,size_t biasesBytes,void *output,size_t outputBytes) {
    Layout l(rows);
    if(!rows||rows>Rows||!codes||!scales||!biases||!output||
        codesBytes!=size_t(rows)*Columns||scalesBytes!=size_t(rows)*64||
        biasesBytes!=scalesBytes||outputBytes!=l.bytes)return false;
    auto *dst=static_cast<uint8_t*>(output);
    auto *src=static_cast<const uint8_t*>(codes);
    for(size_t tile=0;tile<l.paddedRows/4;tile++) {
        for(uint32_t block=0;block<8;block++)for(uint32_t lane=0;lane<32;lane++)for(uint32_t r=0;r<4;r++) {
            uint32_t row=uint32_t(tile*4+r),column=block*256+lane*8;
            auto *at=dst+l.code(row,column);
            if(row<rows)std::memcpy(at,src+size_t(row)*Columns+column,8);
            else std::memset(at,0,8);
        }
        for(uint32_t g=0;g<32;g++)for(uint32_t r=0;r<4;r++) {
            uint32_t row=uint32_t(tile*4+r);size_t at=l.parameter(row,g)*2;
            if(row<rows) {
                std::memcpy(dst+l.scalesOffset+at,static_cast<const uint8_t*>(scales)+(size_t(row)*32+g)*2,2);
                std::memcpy(dst+l.biasesOffset+at,static_cast<const uint8_t*>(biases)+(size_t(row)*32+g)*2,2);
            } else {
                std::memset(dst+l.scalesOffset+at,0,2);std::memset(dst+l.biasesOffset+at,0,2);
            }
        }
    }
    return true;
}
}
