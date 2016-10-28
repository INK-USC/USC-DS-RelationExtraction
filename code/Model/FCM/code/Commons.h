//
//  Commons.h
//  RE_FCT
//
//  Created by gflfof gflfof on 14-8-30.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef RE_FCT_Commons_h
#define RE_FCT_Commons_h

#define MAX_STRING 256
#define EXP_TABLE_SIZE 1000
//#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

//#define MAX_EXP 10
//#define MIN_EXP -10

struct word_info {
    char codelen;
    int point[MAX_CODE_LENGTH];
    char code[MAX_CODE_LENGTH];
};


#endif
