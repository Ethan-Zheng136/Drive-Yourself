"""
Chain of Thought (CoT) prompts for driving scenario reasoning.
This module contains shared prompt templates used across different dataset processors.
"""


def get_cot_reasoning_prompt(fut_ego_action: str) -> dict:
    """
    Generate the Chain of Thought reasoning prompt for driving scenarios.
    
    Args:
        fut_ego_action: The ground truth future ego action (e.g., "move forward with an acceleration")
    
    Returns:
        A dictionary with "type" and "text" keys containing the CoT reasoning prompt
    """
    prompt_text = (
        "Based on the above camera images and ego vehicle states, please predict:" +
        "1. Scene Description: Describe the driving scene according to the movements of other cars or pedestrians, lane markings, traffic lights (if available), and other vehicles' indicator lights (if available)." +
        "2. Critical Object Description: What road users or map elements should you pay attention to? List two or three of them, specify their locations in the image, and provide a short description of what they are doing, what they will do, and why they are important to you." +
        "3. Reasoning on Intent: Based on the movement of other cars and pedestrians, traffic lights (if available), lane markings, and ego vehicle current movement states, describe the desired intent of the ego car." +
        "4. Given the **Best Driving Action** of the ego car uses a combination of"
        "- Lateral actions (choose exactly one):" +
        "  [move forward, turn left, change lane to left, turn right, change lane to right]" +
        "- Longitudinal actions (choose exactly one):" +
        "  [stop, a deceleration to zero, a constant speed, a quick deceleration, a deceleration, a quick acceleration, an acceleration]" +
        
        f"Hint: the ground truth of the best driving action is {fut_ego_action} in this scenario. You need to ensure your reasoning processing and the final result align with it."

        "Below are five examples demonstrating how you might structure your answers (questions and answers).\n\n" +
        
        "----------------------------------------\n" +
        "**Example 1**\n" +
        "Q (Scene Description):\n" +
        "\"There are many barriers, two cars, and many pedestrians behind the ego car. " +
        "There are many barriers, one white SUV, one bicycle and one traffic cone to the front of the ego car. " +
        "The maps include a straight or left turn white arrow to the front of the ego vehicle, " +
        "and a straight or right turn white arrow to the right front of the ego vehicle. It is a daytime scene, and the road is slightly wet\"\n\n" +
        "Q (Critical Object Description):\n" +
        "\"The moving status of **White SUV** is turning right, because the SUV has the right-turn signal on.\"\n\n" +
        "Q (Reasoning on Intent):\n" +
        "\"Firstly, notice **Straight or left turn white arrow**, which marks the lane the ego vehicle drives on, meaning the ego vehicle should follow the guidance." +
        "Secondly, notice **Straight or right turn white arrow**, which marks the other lane, and the ego vehicle should not follow." +
        "Thirdly, notice **White SUV**. It is moving ahead with the right-turn indicator on." +
        "Finally, notice **Bicycle**, which is stop in front of the ego vehicle." +
        "Meanwhile, the ego vehicle is stopped in the past 4s based on its past behavior stop, and its velocity is close to zero. The driving instruction is to turn left." +
        "Thus, the ego vehicle should turn left with an acceleration and keep .\"\n\n" +
        "Q (Best Driving Action):\n" +
        "\"turn left with an acceleration\"\n" +
        "----------------------------------------\n\n" +
        
        "----------------------------------------\n" +
        "**Example 2**\n" +
        "Q (Scene Description):\n" +
        "\"There are many cars, one bus, and two traffic cones behind the ego car. There are many cars in front of the ego car, " +
        "including a black car and a white sedan. Maps include a straight and a left-turn arrow to the front of the ego vehicle. " +
        "And there is a STOP sign in front of the ego vehicle. It's a bright sunny day.\"\n\n" +
        "Q (Critical Object Description):\n" +
        "\"The moving status of **Black car** is stationary (brake lights on). " +
        "The moving status of **White sedan** is stationary (brake lights on).\"\n\n" +
        "Q (Reasoning on Intent):\n" +
        "\"Firstly, notice that **Black car**. The object is stationary in the left-turn lane rather than the lane the vehicle is driving on. " +
        "Secondly, notice that **Left-turn arrow**, which marks the other lane, and the ego vehicle should not follow." +
        "Thirdly, notice that **STOP sign**, which means the vehicle should stop and then continue driving after confirming it's safe." +
        "Although the left lane has available space for the ego vehicle to change lanes, the driving instruction is to keep forward." +
        "Meanwhile, the ego vehicle has decelerated to stop based on its behavior in the past 4 seconds, and the ego vehicle is stopped now because its velocity is close to zero." +
        "The environment is safe for the ego vehicle to go, thus, the ego vehicle should remain cautious and move forward with a quick acceleration. + \n\n" +
        "Q (Best Driving Action):\n" +
        "\"move forward with a quick acceleration\"\n" +
        "----------------------------------------\n\n" +
        
        "----------------------------------------\n" +
        "**Example 3**\n" +
        "Q (Scene Description):\n" +
        "\"There are three pedestrians and a silver sedan in front of the ego car. " +
        "There are many pedestrians, one truck, one white SUV, one bus, and one traffic cone behind the ego car. " +
        "The traffic light in front of the ego vehicle is red. It is a daytime scene, and the weather is foggy.\"\n\n" +
        "Q (Critical Object Description):\n" +
        "\"The moving status of **Pedestrian** is keep going straight. The moving status of **White SUV** " +
        "is braking gently to come to a stop (brake lights on). The moving status of **Silver sedan** is stop with brake lights on.\"\n\n" +
        "Q (Reasoning on Intent):\n" +
        "\"Firstly, notice that **Silver sedan**. The object is in the lane ego vehicle drives on. " +
        "Secondly, notice that **Red light**. The object is a traffic sign, meaning all objects in this lane need to wait." +
        "Thirdly, notice that **Pedestrian**, they are going ahead, and have no intention to use sthe vehicle lane." +
        "Meanwhile, the ego vehicle is moving forward with a constant speed in the past 4s." +
        "Thus, the ego vehicle should remain cautious and decelerate to zero to keep a safe distance from the stationary **Silver sedan**." +
        "The vehicle also needs to be careful in the fog, which leads to low visibility.\"\n\n" +
        "Q (Best Driving Action):\n" +
        "\"move forward with a deceleration to zero\"\n" +
        "----------------------------------------\n\n" +
        
        "----------------------------------------\n" +
        "**Example 4**\n" +
        "Q (Scene Description):\n" +
        "\"The scene is at night with rain, we need to identify the vehicle position light and traffic light carefully. " +
        "There is a construction vehicle, two pedestrians, a white truck, and three traffic cones in front of the ego car." +
        "There is one truck behind the ego car. There are some lights ahead, but the traffic light status is unclear and still far away. " +
        "There is a red traffic light behind the ego car." +
        "Q (Critical Object Description):\n" +
        "\"The moving status of **White truck** is turning right. Its position light is on (general brightness, it's not brake light), because the scene is at night." +
        "\"The moving status of **Construction vehicles** is stationary (brake lights on, it is brighter than the position light)." +
        "Q (Reasoning on Intent):\n" +
        "\"Firstly, notice **Red light** behind the vehicle. **ego vehicle does not need to follow the traffic sign or traffic light in the back view**, which are set for the opposite lane." +
        "Secondly, notice **White truck**, which will turn right on the right-turn lane. The right-turn lane is on the right side of the ego vehicle. " +
        "Thirdly, notice **Unclear traffic light** in front. There are some lights in the night scene, and the ego vehicle does not need to follow the unclear traffic light." +
        "The ego vehicle is moving on the go straight lane in the past 4s, but the driving command is turn right now. Meanwhile, no vehicle is in the right behind" +
        "Thus, the ego vehicle should change the lane to the right, and safely go into the right-turn lane and follow the White truck.\"\n\n" +
        "Q (Best Driving Action):\n" +
        "\"change lane to right with an acceleration\"\n" +
        "----------------------------------------\n\n" +
        
        "----------------------------------------\n" +
        "**Example 5**\n" +
        "Q (Scene Description):\n" +
        "\"The scene is at night. The ego vehicle is approaching an intersection where a pedestrian is on the sidewalk in front right of the vehicle, and some vehicles are parked on the roadside. " +
        "There is another white vehicle in the left front of the vehicle, which is driving on the opposite lane. Meanwhile, there are no traffic lights and no traffic signs." +
        "Q (Critical Object Description):\n" +
        "\"The moving status of **Pedestrian** is walking on the sidewalk, and does not show the intent to cross the road." +
        "\"The moving status of **Parking vehicles** is stationary, and do not move anymore." +
        "Q (Reasoning on Intent):\n" +
        "\"Firstly, notice **Pedestrian** in front of the vehicle, and they will keep walking on the sidewalk." +
        "Secondly, notice **Parking vehicles**, and the ego vehicle should keep a lateral distance from them. " +
        "Thirdly, notice **White vehicle** in the opposite lane, and ego vehicle should keep a safe lateral distance from it." +
        "The ego vehicle is moving on the go straight lane in the past 4s, and no traffic signs or lights need to be followed." +
        "Thus, the ego vehicle should keep moving forward with a constant speed.\"\n\n" +
        "Q (Best Driving Action):\n" +
        "\"move forward with a constant speed\"\n" +
        "----------------------------------------\n\n" +
        
        "Finally, when providing your answer for the current scene, please follow a similar structure " +
        "(scene description, critical object description, reasoning on intent, and best driving action). "
    )
    
    return {
        "type": "text",
        "text": prompt_text
    }

