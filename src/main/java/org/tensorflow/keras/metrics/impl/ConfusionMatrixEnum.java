/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.metrics.impl;

/**
 *
 * @author Jim Clarke
 */
public enum ConfusionMatrixEnum {
    TRUE_POSITIVES("tp"),
    FALSE_POSITIVES("fp"),
    TRUE_NEGATIVES("tn"),
    FALSE_NEGATIVES("fn");
    
    private final String abbrev;
    
    private ConfusionMatrixEnum(String abbrev){
        this.abbrev = abbrev;
    }
    
    public String getAbbreviation() {
        return abbrev;
    }
    
    public static ConfusionMatrixEnum get(String item) {
        ConfusionMatrixEnum cm = valueOf(item.toUpperCase());
        if(cm == null) {
            for(ConfusionMatrixEnum m : values()) {
                if(m.getAbbreviation().equals(item.toLowerCase())) {
                    return m;
                }
            }
        }
        return null;
    }
    
    
    
}
