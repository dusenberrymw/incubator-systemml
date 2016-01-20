/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.parser;

import java.util.HashMap;

import org.apache.sysml.parser.dml.DMLParserWrapper;
import org.apache.sysml.parser.pydml.PyDMLParserWrapper;

/**
 * Base class for all dml parsers in order to make the various compilation chains
 * independent of the used parser.
 */
public abstract class AParserWrapper 
{
	//global parser configuration dml/pydml:
	//1) skip errors on unspecified args (modified by mlcontext / jmlc)
	public static boolean IGNORE_UNSPECIFIED_ARGS = false; 
	
	/**
	 * 
	 * @param fileName
	 * @param dmlScript
	 * @param argVals
	 * @return
	 * @throws ParseException
	 */
	public abstract DMLProgram parse(String fileName, String dmlScript, HashMap<String,String> argVals) 
		throws ParseException;
	
	
	/**
	 * Factory method for creating instances of AParserWrapper, for
	 * simplificy fused with the abstract class.
	 * 
	 * @param pydml
	 * @return
	 */
	public static AParserWrapper createParser(boolean pydml)
	{
		AParserWrapper ret = null;
		
		//create the parser instance
		if( pydml )
			ret = new PyDMLParserWrapper();
		else
			ret = new DMLParserWrapper();
		
		return ret;
	}
}
