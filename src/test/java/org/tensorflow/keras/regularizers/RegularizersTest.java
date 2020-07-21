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
package org.tensorflow.keras.regularizers;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.op.Ops;

/**
 *
 * @author jbclarke
 */
public class RegularizersTest extends CommonTest {
    
    public RegularizersTest() {
    }
    
    @BeforeAll
    public static void setUpClass() {
    }
    
    @AfterAll
    public static void tearDownClass() {
    }
    
    @BeforeEach
    public void setUp() {
    }
    
    @AfterEach
    public void tearDown() {
    }

    /**
     * Test of get method, of class Regularizers.
     */
    @Test
    public void testGet_Object_String() {
        String initializerFunction = "l1_l2";
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Regularizer result = Regularizers.get(tf, initializerFunction);
            assertNotNull(result);
            assertTrue(result instanceof l1_l2);
        }
    }

    /**
     * Test of get method, of class Regularizers.
     */
    @Test
    public void testGet_Object_Lambda() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Regularizer result = Regularizers.get(tf, ops -> new L1L2(ops, 0.2f, 0.3f));
            assertNotNull(result);
            assertTrue(result instanceof L1L2);
            assertEquals(0.2f, ((L1L2)result).getL1());
            assertEquals(0.3f, ((L1L2)result).getL2());
        }
    }

    /**
     * Test of get method, of class Regularizers.
     */
    @Test
    public void testGet_Object_Class() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Regularizer result = Regularizers.get(tf, L2.class);
            assertNotNull(result);
            assertTrue(result instanceof L2);
            L2 instance = (L2)result;
            assertEquals(Regularizer.DEFAULT_REGULARIZATION_PENALTY, instance.getL2());
            assertNull(instance.getL1());
        }
    }

    /**
     * Test of get method, of class Regularizers.
     */
    @Test
    public void testGet_Object_Initializer() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Function<Ops, Regularizer> regularizerFunction = ops -> new L1(ops);
            Regularizer result = Regularizers.get(tf, regularizerFunction);
            assertNotNull(result);
            assertTrue(result instanceof L1);
        }
    }

    /**
     * Test of get method, of class Regularizers.
     */
    @Test
    public void testGet_Object_Unknown() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            String initializerFunction = "bogus";
            Regularizer result = Regularizers.get(tf, initializerFunction);
            assertNull(result);
        }
    }

    /**
     * Test of get method, of class Regularizers.
     */
    @Test
    public void testGet_Object_Map() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Map<String, Function<Ops, Regularizer>> custom_functions = new HashMap<>();
            custom_functions.put("foobar", ops -> new l1_l2(ops));
            String initializerFunction = "foobar";
            Regularizer result = Regularizers.get(tf, initializerFunction, custom_functions);
            assertNotNull(result);
            assertTrue(result instanceof l1_l2);
        }
    }
     /**
     * Test of l1_l2 method, of class Regularizers.
     */
    @Test
    public void testL1L2_Ops() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            L1L2 instance = Regularizers.L1L2(tf);
            assertNull(instance.getL1());
            assertNull( instance.getL2());
        }
    }

    /**
     * Test of l1_l2 method, of class Regularizers.
     */
    @Test
    public void testL1L2_3args() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            L1L2 instance = Regularizers.L1L2(tf, 0.05f, 0.6f);
            assertEquals( 0.05f, instance.getL1());
            assertEquals(0.6f, instance.getL2());
        }
    }
    
    /**
     * Test of l1_l2 method, of class Regularizers.
     */
    @Test
    public void testL1_l2_Ops() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            l1_l2 instance = Regularizers.l1_l2(tf);
            assertEquals(Regularizer.DEFAULT_REGULARIZATION_PENALTY, instance.getL1());
            assertEquals(Regularizer.DEFAULT_REGULARIZATION_PENALTY, instance.getL2());
        }
    }

    /**
     * Test of l1_l2 method, of class Regularizers.
     */
    @Test
    public void testL1_l2_3args() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            l1_l2 instance = Regularizers.l1_l2(tf, 0.05f, 0.6f);
            assertEquals( 0.05f, instance.getL1());
            assertEquals(0.6f, instance.getL2());
        }
    }

    /**
     * Test of l1 method, of class Regularizers.
     */
    @Test
    public void testL1_Ops() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            L1 instance = Regularizers.l1(tf);
            assertEquals( Regularizer.DEFAULT_REGULARIZATION_PENALTY, instance.getL1());
            assertNull(instance.getL2());
        }
    }

    /**
     * Test of l1 method, of class Regularizers.
     */
    @Test
    public void testL1_Ops_float() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            float l1 = 0.001F;
            L1 instance = Regularizers.l1(tf, l1);
            assertEquals( l1, instance.getL1());
            assertNull(instance.getL2());
        }
    }

    /**
     * Test of l2 method, of class Regularizers.
     */
    @Test
    public void testL2_Ops() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            L2 instance = Regularizers.l2(tf);
            assertEquals( Regularizer.DEFAULT_REGULARIZATION_PENALTY, instance.getL2());
            assertNull(instance.getL1());
        }
    }

    /**
     * Test of l2 method, of class Regularizers.
     */
    @Test
    public void testL2_Ops_float() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            float l2 = 0.002F;
            L2 instance = Regularizers.l2(tf, l2);
            assertEquals( l2,  instance.getL2());
            assertNull(instance.getL1());
        }
    }
    
}
