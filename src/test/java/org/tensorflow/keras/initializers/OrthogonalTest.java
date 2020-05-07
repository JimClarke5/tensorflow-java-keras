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
package org.tensorflow.keras.initializers;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.PrintUtils;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.tools.buffer.DataBuffers;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;

/**
 *
 * @author Jim Clarke
 */
public class OrthogonalTest {
    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;
    private static final long SEED = 1000L;
    private static final double GAIN_VALUE = 1.0;
    
    public OrthogonalTest() {
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
     * Test of getConfig method, of class Orthogonal.
     */
    @Test
    public void testGetConfig() {
        System.out.println("getConfig");
        Map<String, Object> config = new HashMap<>();
        config.put(Orthogonal.GAIN_KEY, GAIN_VALUE);
        config.put(Orthogonal.SEED_KEY, SEED);    
        Orthogonal instance = new Orthogonal(GAIN_VALUE,  SEED);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }
    
    /**
     * Test of getConfig method, of class Orthogonal.
     */
    @Test
    public void testConfigCTORMap() {
        System.out.println("ctor Map");
        Map<String, Object> config = new HashMap<>();
        config.put(Orthogonal.GAIN_KEY, GAIN_VALUE);
        config.put(Orthogonal.SEED_KEY, SEED);    
        Orthogonal instance = new Orthogonal(config);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }

    /**
     * Test of call method, of class Orthogonal.
     */
    @Test
    public void testCall_Int() {
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(10,10);
            Orthogonal<TInt32> instance = 
                    new Orthogonal(GAIN_VALUE,  SEED);
            Operand<TInt32> operand = instance.call(tf, tf.constant(shape.asArray()),  TInt32.DTYPE);
            fail("Should jave thown assertion on Integer type");
        }catch (AssertionError expected) {
            
        }
    }
    /**
     * Test of call method, of class Orthogonal.
     */
    @Test
    public void testCall_Float() {
        System.out.println("callFloat");
        float[] actual = new float[10*10];
        float[] expected = {  
           -0.3097564F, -0.11214957F, -0.04083291F, -0.24071707F,  0.29931748F,  0.4461752F,
                -0.16319607F, -0.30204326F, -0.26093683F,  0.59770143F,
            0.15418966F,  0.50748324F, -0.03822303F, -0.59814125F,  0.11034431F, -0.01813965F,
                -0.21199228F, -0.04033701F, -0.40765563F, -0.36632827F,
            0.10572237F,  0.27673772F, -0.00941799F,  0.07603773F,  0.48299354F,  0.37719437F,
                 0.65557724F,  0.31341612F,  0.04323304F, -0.03049367F,
           -0.00511622F, -0.30234647F, -0.24784878F, -0.27694383F, -0.6077379F,  0.40848815F,
                 0.40706915F, -0.0732277F, -0.16744994F, -0.18739915F,
           -0.151793F, -0.21273288F,  0.24265847F, -0.00964088F,  0.25967413F,  0.40649366F,
                -0.20693113F, -0.3185814F,  0.38828942F, -0.5873469F,
           -0.48195702F,  0.32218578F, -0.29953587F,  0.00851173F,  0.01569128F, -0.33701414F,
                 0.36372715F, -0.54230285F,  0.17351612F, -0.06162076F,
           -0.2438229F,  0.35682017F,  0.7260855F,  0.24974659F, -0.34703425F,  0.14939374F,
                 0.09953088F, -0.08766067F, -0.25020337F,  0.02669237F,
            0.41220927F,  0.4300388F, -0.03955907F, -0.11728173F, -0.2787032F,  0.26550797F,
                -0.11485924F, -0.19093868F,  0.5791758F,  0.3107499F,
           -0.46279088F, -0.04041088F,  0.23238355F, -0.5590758F, -0.07460429F, -0.13264497F,
                 0.04314278F,  0.47426552F,  0.39604855F,  0.10401782F,
           -0.41256273F,  0.31454724F, -0.45164356F,  0.33607012F, -0.1557368F,  0.31974515F,
                 -0.3645014F,  0.37268594F, -0.00656797F, -0.12504758F
        };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(10,10);
            Orthogonal<TFloat32> instance = 
                    new Orthogonal(GAIN_VALUE,  SEED);
            Operand<TFloat32> operand = instance.call(tf, tf.constant(shape.asArray()),  TFloat32.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTFloat32(operand.asTensor());
            assertArrayEquals(actual,expected, EPSILON_F);
        }
    }
    
    /**
     * Test of call method, of class Orthogonal.
     */
    @Test
    public void testCall_Double() {
        System.out.println("callDouble");
        double[] actual = new double[10*10];
        double[] expected = {  
           0.4852166440161694, -0.4290733656784607, 0.09147039077482466, -0.3033533647665251,
           -0.13422222791377508, -0.3129540993206184, 0.183062569636896, -0.0797586175921162,
           0.36040118003516713, -0.4408543207576007, -0.2732620793726008, -0.031650254737601205,
           -0.487642998274466, 0.18503560531435395, -0.14802624287521232, 0.11682409149136887,
           0.43002930688897106, 0.39878194544024825, -0.10095741255484561, -0.5124333590595215,
           0.18311274512293216, 0.14142110936521135, -0.21143499395594853, -0.11919423920003563,
           0.24017159729187723, -0.2593437441508134, 0.667745346902609, -0.35315808322575254,
           -0.3404386389145398, 0.2758862960934774, 0.07139824569700268, 0.09448264380916259,
           0.465791321612319, 0.4543680210644348, 0.5148494317797793, -0.1548002078084767,
           0.2763767527316248, 0.37222851387188227, 0.2398314168577794, -0.03275882219483219,
           0.19185631817009907, 0.05900663337192141, 0.018173647881195746, 0.37339628154719684,
           0.11377436263350496, 0.578439238185625, 0.06494636168027997, -0.5948057813239421,
           0.1107116755187609, -0.319607142429973, 0.2155568630609747, 0.09929282909444799, -0.5490811366582051,
           -0.010954009451688201, 0.11707862431173488, 0.1617550218319554, 0.01106019309067251,
           0.14579444371591477, 0.6518483278305304, 0.3948536518472629, 0.2319871561912634, -0.18238927225826657,
           0.03282551370311214, -0.48208882285440263, 0.46518806000653323, 0.5239030340556176, 
           -0.02248815414434615, 0.3216103558486239, -0.2874388067830515, -0.044661384666030306,
           0.15464707821517193, -0.08187337600211494, 0.3577511581572764, 0.03953488082715882,
           -0.5961789252666962, 0.3822951575732457, 0.4187023892379448, 0.1923143091248148, 
           0.010556064157240419, 0.35474683982006183, 0.643204326887452, -0.07277000873865974,
           -0.22821669120828425, 0.45985896233305346, -0.11635349685972587, -0.12498127959759603,
           -0.2799591321237366, 0.20319311304196724, -0.4071624009218664, 0.053248119820197976,
           0.2766685450718409, 0.8528551980781793, 0.0959402007341447, -0.2609469621757593,
           -0.15906257638032784, -0.013734816737670838, -0.02756903693269743, 0.12075359886144169,
           0.028705024822326536, -0.27774030642345227

        };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(10,10);
            Orthogonal<TFloat64> instance = 
                    new Orthogonal(GAIN_VALUE,  SEED);
            Operand<TFloat64> operand = instance.call(tf, tf.constant(shape.asArray()),  TFloat64.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTFloat64(operand.asTensor());
            assertArrayEquals(actual,expected, EPSILON);
        }
    }
    
}
