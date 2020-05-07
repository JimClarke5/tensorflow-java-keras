/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.utils;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import static org.junit.jupiter.api.Assertions.*;

/**
 *
 * @author Jim Clarke
 */
public class OptionsTest {
    
    public OptionsTest() {
    }

    @org.junit.jupiter.api.BeforeAll
    public static void setUpClass() throws Exception {
    }

    @org.junit.jupiter.api.AfterAll
    public static void tearDownClass() throws Exception {
    }

    @org.junit.jupiter.api.BeforeEach
    public void setUp() throws Exception {
    }

    @org.junit.jupiter.api.AfterEach
    public void tearDown() throws Exception {
    }
    
    /**
     * Test of create method, of class Options.
     */
    @org.junit.jupiter.api.Test
    public void testCreate_0args() {
        System.out.println("create");
        Options expResult = new Options();
        Options result = Options.create();
        assertEquals(expResult, result);
    }

    /**
     * Test of create method, of class Options.
     */
    @org.junit.jupiter.api.Test
    public void testCreate_Map() {
        System.out.println("create");
        Map<String, Object> sourceMap = new HashMap<>();
        sourceMap.put("foo", 2);
        Options expResult = new Options();
        expResult.addOption("foo", 2);
        Options result = Options.create(sourceMap);
        assertEquals(expResult, result);
    }

    /**
     * Test of create method, of class Options.
     */
    @org.junit.jupiter.api.Test
    public void testCreate_String() {
        System.out.println("create");
        String jsonString = "{ \"foo\" : 2}";
        Options expResult = new Options();
        expResult.addOption("foo", 2);
        Options result = Options.create(jsonString);
        assertEquals(expResult, result);
    }

    /**
     * Test of pop method, of class Options.
     */
    @org.junit.jupiter.api.Test
    public void testPop_String() {
        System.out.println("pop");
        String jsonString = "{ \"foo\" : 2}";
        String key = "foo";
        Options instance = new Options(jsonString);
        Object expResult = 2;
        Object result = instance.pop(key);
        assertEquals(expResult, result);
    }

    /**
     * Test of pop method, of class Options.
     */
    @org.junit.jupiter.api.Test
    public void testPop_String_Object() {
        System.out.println("pop");
        String jsonString = "{ \"foo\" : 2}";
        String key = "foo";
        Object defaultValue = 3;
        Options instance = new Options(jsonString);
        Object expResult = 2;
        Object result = instance.pop(key, defaultValue);
        assertEquals(expResult, result);
        expResult = 3;
        result = instance.pop(key, defaultValue);
        assertEquals(expResult, result);
    }

    /**
     * Test of validate method, of class Options.
     */
    @org.junit.jupiter.api.Test
    public void testValidate_StringArr() {
        System.out.println("validate");
        String[] allowed_keys = { "one", "two", "three" };
        Options instance = new Options();
        instance.validate(allowed_keys);
        instance.addOption("foo", "bar");
        try {
            instance.validate(allowed_keys);
            fail("Expected an IllegalArgumentException");
        }catch (IllegalArgumentException ex) {
            // Expected
        }
    }

    /**
     * Test of validate method, of class Options.
     */
    @org.junit.jupiter.api.Test
    public void testValidate_Set() {
        System.out.println("validate");
        String[] allowed_keys = {"one", "two", "three"};
        Options instance = new Options();
        instance.validate(allowed_keys);
        instance.addOption("foo", "bar");
        try {
            instance.validate(allowed_keys);
            fail("Expected an IllegalArgumentException");
        }catch (IllegalArgumentException ex) {
            // Expected
        }
    }

    /**
     * Test of build method, of class Options.
     */
    @org.junit.jupiter.api.Test
    public void testBuild() {
        System.out.println("build");
        Options instance = new Options();
        Options expResult = instance;
        Options result = instance.build();
        assertEquals(expResult, result);
    }

    /**
     * Test of getMap method, of class Options.
     */
    @org.junit.jupiter.api.Test
    public void testGetMap() {
        System.out.println("getMap");
        Options instance = new Options();
        Map<String, Object> expResult = new HashMap<>();
        Map<String, Object> result = instance.getMap();
        assertEquals(expResult, result);
    }

    /**
     * Test of addOption method, of class Options.
     */
    @org.junit.jupiter.api.Test
    public void testAddOption() {
        System.out.println("addOption");
        String key = "foo";
        Object value = "bar";
        Options instance = new Options();
        Options expResult = instance;
        Options result = instance.addOption(key, value);
        assertSame(expResult, result);
        
        Object resultValue = instance.get(key);
        assertEquals(value, resultValue);
    }

    /**
     * Test of getOrDefault method, of class Options.
     */
    @org.junit.jupiter.api.Test
    public void testGetOrDefault() {
        System.out.println("getOrDefault");
        String key = "foo";
        Object defaultValue = "bar";
        Options instance = new Options();
        Object expResult = defaultValue;
        Object result = instance.getOrDefault(key, defaultValue);
        assertEquals(expResult, result);
    }

    /**
     * Test of get method, of class Options.
     */
    @org.junit.jupiter.api.Test
    public void testGet() {
        System.out.println("get");
        String key = "foo";
        Object value = "bar";
        Options instance = new Options();
        Object expResult = null;
        Object result = instance.get(key);
        assertNull(result);
        instance.addOption(key, value);
        expResult = value;
        result = instance.get(key);
        assertEquals(expResult, result);
    }
    
}
