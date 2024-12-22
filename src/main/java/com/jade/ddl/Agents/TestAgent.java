package com.jade.ddl.Agents;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jade.core.AID;
import jade.core.Agent;
import jade.core.behaviours.CyclicBehaviour;
import jade.lang.acl.ACLMessage;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class TestAgent extends Agent {

    @Override
    protected void setup() {
        System.out.println(getLocalName() + " is ready to perform testing.");

        // Add behavior to listen for testing requests
        addBehaviour(new CyclicBehaviour() {
            @Override
            public void action() {
                // Wait for messages
                ACLMessage msg = receive();
                if (msg != null) {
                    try {
                        // Check if it's a testing request
                        if (msg.getPerformative() == ACLMessage.REQUEST && msg.getContent().equalsIgnoreCase("Test The Model")) {
                            System.out.println(getLocalName() + ": Received a testing request from " + msg.getSender().getLocalName());

                            // Perform testing task
                            ArrayList<Double> testResults = performTesting();
                            System.out.println("Result :: \n"
                                    +"accuracy :"+testResults.get(0)+"\n"
                                    +"precision :"+testResults.get(1)+"\n"
                                    +"recall :"+testResults.get(2)+"\n"
                                    +"f1 score :"+testResults.get(3));

                            // Send response to coordinator
                            ACLMessage reply = new ACLMessage(ACLMessage.INFORM);
                            reply.addReceiver(new AID("Coordinator",AID.ISLOCALNAME));


                            if (testResults.get(0) <= 80){
                                System.out.println("Test Decision :: Bad result");
                                reply.setContent("Train again");
                            }else{
                                System.out.println("Test Decision :: Good result");
                                reply.setContent("Good");
                            }
                            send(reply);




                            System.out.println(getLocalName() + ": Testing completed and results sent to " + msg.getSender().getLocalName());
                        } else {
                            System.out.println(getLocalName() + ": Received an unrecognized message.");
                        }
                    } catch (Exception e) {
                        System.err.println(getLocalName() + ": Error during testing - " + e.getMessage());
                    }
                } else {
                    block(); // No messages, block to wait for new messages
                }
            }
        });
    }

    private ArrayList<Double> performTesting() {
        Map<String, Object> resultMap = new HashMap<>();
        ArrayList<Double> resultArr = new ArrayList<>();

        try {
            // Create connection to the server endpoint
            URL url = new URL("http://localhost:5000/test");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");

            // Check response code
            int responseCode = conn.getResponseCode();
            if (responseCode == 200) {
                // Read the response from the server
                BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                StringBuilder response = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    response.append(line);
                }
                reader.close();

                // Parse JSON response
                ObjectMapper mapper = new ObjectMapper();
                JsonNode jsonResponse = mapper.readTree(response.toString());

                // Extract values from the JSON response
                String status = jsonResponse.get("status").asText();

                if ("success".equalsIgnoreCase(status)) {
                    JsonNode result = jsonResponse.get("result");

                    resultArr.add(result.get("accuracy").asDouble());
                    resultArr.add(result.get("precision").asDouble());
                    resultArr.add(result.get("recall").asDouble());
                    resultArr.add(result.get("f1_score").asDouble());
                } else {
                    resultMap.put("error", "Server responded with status: " + status);
                }
            } else {
                resultMap.put("error", "Server responded with error code: " + responseCode);
            }
        } catch (Exception e) {
            resultMap.put("error", "An error occurred during testing: " + e.getMessage());
        }

        return resultArr;
    }

}
