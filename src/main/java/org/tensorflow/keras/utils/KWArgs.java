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
package org.tensorflow.keras.utils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import org.json.JSONObject;

/**
 * class to handle python kwargs
 * @author Jim Clarke
 */
public class KWArgs {
    private final Map<String, Object> map = new HashMap();
    
    /**
     * create an kwargs instance for builder pattern.
     * for example: <code>KWArgs.create().add("foo", 2).build();</code>
     * 
     * @return the KWArgs object
     */
    public static KWArgs create() {
        return new KWArgs();
    }
    
    /**
     * create an kwargs instance for builder pattern.
     * for example: <code>KWArgs.create(initMap).add("foo", 2).build();</code>
     * 
     * @param sourceMap a Map instance to initialize this KWArgs instance
     * @return the KWArgs object
     */
    public static KWArgs create(Map<String, Object> sourceMap) {
        return new KWArgs(sourceMap);
    }
    
    /**
     * create an kwargs instance for builder pattern.
     * for example: <code>KWArgs.create("{\"foo\" : 2 }").build();</code>
     * 
     * @param jsonString the JSON Object as a string to initialize the KWArgs 
     * @return the KWArgs object
     */
    public static KWArgs create(String jsonString) {
        return new KWArgs(jsonString);
    }
    
    /**
     * Default constructor
     */
    public KWArgs() {
    }
    
    /**
     * Constructs the KWArgs 
     * 
     * @param sourceMap a Map instance to initialize this KWArgs instance
     */
    public KWArgs(Map<String, Object> sourceMap) {
        for(String key : sourceMap.keySet()) {
            this.map.put(key, sourceMap.get(key));
        }
    }
    
    /**
     * Constructs the KWArgs 
     * @param jsonString the JSON Object as a string to initialize the KWArgs 
     */
    public KWArgs(String jsonString) {
        assert(jsonString != null);
        this.initializeFromJson(new JSONObject(jsonString));
    }
    
    /**
     * Constructs the KWArgs 
     * @param jsonObject a JSON Object to initialize the KWArgs 
     */
    public KWArgs(JSONObject jsonObject) {
        assert(jsonObject != null);
       this.initializeFromJson(jsonObject);
    }
    
    /**
     * populate this object from a JSONObject
     * 
     * @param jsonObject  the JSON Object to initialize the KWArgs 
     */
    private void initializeFromJson(JSONObject jsonObject) {
        jsonObject.keySet().forEach((s) -> {
            map.put(s, jsonObject.get(s));
        });
    }
        
    
    /**
     * Pop an KWArgs value
     * 
     * @param key the key 
     * @return the value or null if the key does not exist. Removes the option.
     */
    public Object pop(String key) {
        return this.map.remove(key);
    }
    
    /**
     * Pop an KWArgs value, if it exists
     * @param key the key 
     * @param defaultValue the default value
     * @return the value or defaultValue if the key does not exist. Removes the option.
     */
    public Object pop( String key, Object defaultValue) {
        return this.map.containsKey(key)? this.map.remove(key) : defaultValue;
    }
    
    /**
     * Validate that this kwargs only contains the allowed keys
     * 
     * @param allowed_keys the allowed keys
     * @throws IllegalArgumentException if there are unallowed keys in the kwargs.
     */
    public  void validate( String... allowed_keys ) {
        Set<String> theSet  = new HashSet( Arrays.asList(allowed_keys) );
        validate(theSet);
    }
    
    /**
     * Validate that this kwargs only contains the allowed keys
     * 
     * @param allowed_keys the allowed keys
     * @throws IllegalArgumentException if there are unallowed keys in the kwargs.
     */
    public void validate(Set<String> allowed_keys ) {
        this.map.keySet().stream().filter((key) -> (!allowed_keys.contains(key))).forEachOrdered((key) -> {
            throw new IllegalArgumentException("Unexpected keyword argument passed to optimizer: " + key);
        });
    }
   
    /**
     * builds the kwargs for builder pattern
     * 
     * @return this KWArgs instance
     */
    public KWArgs build() {
        return this;
    }
    
    
    /**
     * Get the underlying map
     * 
     * @return the underlying map
     */
    public Map<String, Object> getMap() {
        return map;
    }
    
    /**
     * Add an option
     * 
     * @param key the option key
     * @param value the option value
     * @return this KWArgs instance
     */
    public KWArgs addOption(String key, Object value) {
        this.map.put(key, value);
        return this;
    }
    
    /**
     * test to see if an option exists
     * 
     * @param option the option to check
     * @return true if the option is defined
     */
    public boolean containsOption(String option) {
        return this.map.containsKey(option);
                
    }
    
    /**
     * get the option value or default if the option key does not exist
     * 
     * @param key the option key
     * @param defaultValue the option value
     * @return the option value or default value if the key does not exist.
     */
    public Object getOrDefault(String key, Object defaultValue) {
        return map.getOrDefault(key, defaultValue);
    }
    
    /**
     * get the option value
     * 
     * @param key key the option key
     * @return the option value or null if the key does not exist.
     */
    public Object get(String key) {
        return map.get(key);
    }
    
    @Override
    public boolean equals(Object other) {
        if(other == null || !(other instanceof KWArgs)) {
            return false;
        }
        KWArgs o = (KWArgs) other;
        return this.map.equals(o.map);
    }

    @Override
    public int hashCode() {
        int hash = 5;
        hash = 71 * hash + Objects.hashCode(this.map);
        return hash;
    }
    
}
