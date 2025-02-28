import os
import cv2 as cv
import numpy as np
import pytesseract
import shutil
import json
from itertools import groupby

class Recognizer:
    __contatore = 0
    __preprocessed_template = False

    __visualize_template_match = False
    __visualize_ORB_matching = False
    __visualize_text_recognition = False

    __custom_languages = {
        "eng": {
            "config":r'--oem 3 --psm 6 -c textord_old_xheight=1 textord_min_xheight=15 textord_max_xheight=30 tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,',
            "lang": r"eng"
        },
        "gop_2400_best": {
            "config":r'--oem 3 --psm 6 -c textord_old_xheight=1 textord_min_xheight=15 textord_max_xheight=30 tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,',
            "lang": r"gop_2400_best"
        },        
        "gop_2400_fast": {
            "config":r'--oem 3 --psm 6 -c textord_old_xheight=1 textord_min_xheight=15 textord_max_xheight=30 tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,',
            "lang": r"gop_2400_fast"
        },
        "gop_4300_best": {
            "config":r'--oem 3 --psm 6 -c textord_old_xheight=1 textord_min_xheight=15 textord_max_xheight=30 tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,',
            "lang": r"gop_4300_best"
        },        
        "gop_4300_fast": {
            "config":r'--oem 3 --psm 6 -c textord_old_xheight=1 textord_min_xheight=15 textord_max_xheight=30 tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,',
            "lang": r"gop_4300_fast"
        },
        "gop_6500_best": {
            "config":r'--oem 3 --psm 6 -c textord_old_xheight=1 textord_min_xheight=15 textord_max_xheight=30 tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,',
            "lang": r"gop_6500_best"
        },        
        "gop_6500_fast": {
            "config":r'--oem 3 --psm 6 -c textord_old_xheight=1 textord_min_xheight=15 textord_max_xheight=30 tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,',
            "lang": r"gop_6500_fast"
        },
    }
    __char_whitelist = set(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,")
    
    def __init__(self, tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        pass

    def __image_preprocessing_template(self, image_path, output_folder, scaling_factor, threshold_val):
        # Read the image
        image =  cv.imread(image_path)
        
        # Convert to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        _, threshold = cv.threshold(gray, threshold_val, 255, cv.THRESH_BINARY)

        # Resize image to half size using cv2.resize
        if scaling_factor!=1 :
            scaled = cv.resize(threshold, None, None, fx=1/scaling_factor, fy=1/scaling_factor, interpolation=cv.INTER_AREA)
        else:
            scaled = threshold
        # Save the image using os.path
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, filename)
        cv.imwrite(output_path, scaled)
        
        return scaled
    
    def __image_preprocessing_text(self, image_path, output_folder):
        # Read the image
        image =  cv.imread(image_path)

        # Convert to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)       

        # Save the image using os.path
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, filename)
        cv.imwrite(output_path, gray)
        
        return

    def __folder_preprocessing(self, folder_path, output_folder, preprocess_function=None, **kwargs):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        
        os.makedirs(output_folder)

        # If no preprocessing function is provided, use __image_preprocessing_text by default
        if preprocess_function is None:
            raise ValueError("Preprocess function not provided")

        for image in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image)
            preprocess_function(image_path, output_folder, **kwargs)

    def __find_layout(self, source_path, question):
        template_paths = question.get("paths")
        if template_paths is None:
            raise ValueError("Question must include a 'template_path' key.")

        confidence_threshold = question.get("confidence_threshold", 0.9)
        overlap_threshold = question.get("overlap_threshold", 0.1)

        # Load the target and template images
        target = cv.imread(source_path, cv.IMREAD_GRAYSCALE)
        
        match_list = []
        for template_path in template_paths:
            template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)

            # Get the dimensions of the template image
            h, w = template.shape

            # Perform template matching
            result = cv.matchTemplate(target, template, cv.TM_CCOEFF_NORMED)

            # Find all the locations where the template matches the target image
            locations = np.where(result >= confidence_threshold)

            # Create candidates list with rectangles and their scores
            for pt in zip(*locations[::-1]):  # Invert x and y coordinates
                x, y = pt
                score = result[y, x]
                match_list.append({
                    "confidence": score,
                    "size": {"width": w, "height": h},
                    "position": {"x": x, "y": y}
                })

        # First filtering level: between objects of the same type
        match_list = self.__remove_duplicates(match_list, confidence_threshold, overlap_threshold)

        return match_list

    def __find_text(self, source_path, position=None, size=None, language="gop_2400_best"):
        match_list = []
        padding = 30

        if language not in self.__custom_languages:
            print(f"Language '{language}' not supported. Supported languages are: {list(self.__custom_languages.keys())}")
            return match_list
        
        source_image  = cv.imread(source_path, cv.IMREAD_GRAYSCALE)
        x_crop, y_crop, w_crop, h_crop = 0, 10, -10, -20
        if position is not None and size is not None:
            x, y = position["x"]+x_crop, position["y"]+y_crop
            w, h = size["width"]+w_crop, size["height"]+h_crop
            source_image = source_image[y:y+h, x:x+w]

        # Set two thresholds (to adjust based on image)
        T_low = 35   # to identify black text
        T_high = 175  # to identify white text

        # Mask for black text: pixels with intensity lower than T_low
        _, mask_black = cv.threshold(source_image, T_low, 255, cv.THRESH_BINARY_INV)
        mask_black = cv.bitwise_not(mask_black)
        mask_black = cv.copyMakeBorder(mask_black, padding, padding, padding, padding, cv.BORDER_CONSTANT, value=255)
        # Mask for white text: pixels with intensity higher than T_high
        _, mask_white = cv.threshold(source_image, T_high, 255, cv.THRESH_BINARY)
        mask_white = cv.bitwise_not(mask_white)
        mask_white = cv.copyMakeBorder(mask_white, padding, padding, padding, padding, cv.BORDER_CONSTANT, value=255)
        # Combine the two masks one under the other
        combined_masks = np.vstack((mask_black, mask_white))

        cv.imshow(f"Combined masks {self.__contatore}", combined_masks)
        self.__contatore += 1

        # Get detailed data from OCR
        data = pytesseract.image_to_data(mask_black, lang=self.__custom_languages[language]["lang"] ,  config=self.__custom_languages[language]["config"], output_type=pytesseract.Output.DICT)
        data2 = pytesseract.image_to_data(mask_white, lang=self.__custom_languages[language]["lang"] ,  config=self.__custom_languages[language]["config"], output_type=pytesseract.Output.DICT)
        # Filter out empty texts and create matches in one pass
        texts   = data["text"] + data2["text"]
        lefts   = data["left"] + data2["left"]
        tops    = data["top"] + data2["top"]
        widths  = data["width"] + data2["width"]
        heights = data["height"] + data2["height"]
        confidences = data["conf"] + data2["conf"]

        # Create match list
        match_list = [
            {
                "type": "text",
                "details": text,
                "position": {"x": left-padding-x_crop, "y": top-padding+y_crop},
                "size": {"width": width, "height": height},
                "confidence": confidence/100
            }
            for text, left, top, width, height, confidence in zip(texts, lefts, tops, widths, heights, confidences)
            if text.strip()
        ]
        
        return match_list

    def __repair_ocr_output(self, entities_list):
        repaired_list = list()
        for entity in entities_list:
            repaired_string = ''.join(char for char in entity["details"] if char in self.__char_whitelist)

            #TODO: Merge entities if they are close to each other in the same line
            #TODO: Create regex to match specific money formats to correct mistakes on recognizing column, points and B or M

            if (len(repaired_string) > 0 
                and (
                    sum(1 for char in repaired_string if char.isalpha()) >= 3
                    or sum(1 for char in repaired_string if char.isdigit()) >= 1
                )
                and entity["size"]["width"] >= 6 
                and 60 >= entity["size"]["height"] > 15):
                    entity["details"] = repaired_string
                    repaired_list.append(entity)
                    print(f"Accepted entity : {repaired_string}\n"
                      f"\twidth:{entity['size']['width']}\n"
                      f"\theight:{entity['size']['height']}\n"
                      f"\tconfidence:{entity['confidence']}\n"
                      f"\tlen:{len(repaired_string)}\n"
                      f"\talpha:{sum(1 for char in repaired_string if char.isalpha())}\n"
                      f"\tnum:{sum(1 for char in repaired_string if char.isdigit())}")
            else:
                print(f"Discarded entity: {repaired_string}\n"
                      f"\twidth:{entity['size']['width']}\n"
                      f"\theight:{entity['size']['height']}\n"
                      f"\tconfidence:{entity['confidence']}\n"
                      f"\tlen:{len(repaired_string)}\n"
                      f"\talpha:{sum(1 for char in repaired_string if char.isalpha())}\n"
                      f"\tnum:{sum(1 for char in repaired_string if char.isdigit())}")

        self.__remove_duplicates(repaired_list, self.__text_overall_confidence_threshold, self.__text_overall_overlap_threshold)
        
        return repaired_list

    def __remove_duplicates(self, entities_list, confidence_threshold, overlap_threshold):

        if len(entities_list) <= 1:
            return entities_list

        # Create a list to store rectangles and scores
        rectangles = []
        scores = []

        for entity in entities_list:
            x, y = entity["position"]["x"], entity["position"]["y"]
            w, h = entity["size"]["width"], entity["size"]["height"]
            rectangles.append([x, y, w, h])
            scores.append(entity["confidence"])

        # Apply Non-Maximum Suppression
        indices = cv.dnn.NMSBoxes(rectangles, scores, score_threshold=confidence_threshold, nms_threshold=overlap_threshold)

        if indices is None or len(indices) == 0:
            # If there are no indices, clear all elements (or leave the list unchanged, depending on needs)
            entities_list.clear()
            return entities_list

        valid_indices = set(indices.flatten())

        # Filter in-place: keep only elements whose index is in valid_indices,
        # adding the "confidence" field from the original scores.
        new_list = []
        for i, entity in enumerate(entities_list):
            if i in valid_indices:
                entity["confidence"] = scores[i]
                new_list.append(entity)

        entities_list.clear()
        entities_list.extend(new_list)
        return entities_list
    
    def __show_entities(self, target_path, entities_list, scaling_factor):
        color_layout=(0, 0, 255)
        color_text=(0, 0, 255)
        target = cv.imread(target_path)
        for object in entities_list:
            x_layout = object["position"]["x"]
            y_layout = object["position"]["y"]
            w_layout = object["size"]["width"]
            h_layout = object["size"]["height"]
            cv.putText(target, object["details"] , (x_layout, y_layout - 10), cv.FONT_HERSHEY_SIMPLEX, 1, color_layout, 4)
            cv.rectangle(target, (x_layout, y_layout), (x_layout + w_layout, y_layout + h_layout), color_layout, 1)
            for text in object.get("content", []):
                x_text = text["position"]["x"]
                y_text = text["position"]["y"]
                w_text = text["size"]["width"]
                h_text = text["size"]["height"]
                cv.putText(target, text["details"], (x_layout + x_text, y_layout + y_text + 10), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 4)
                cv.rectangle(target, (x_layout + x_text, y_layout + y_text), (x_layout + x_text + w_text, y_layout + y_text + h_text), color_text, 1)
        
        target = cv.resize(target, None, fx=1.0/scaling_factor, fy=1.0/scaling_factor)
        cv.imshow("Entities", target)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def __create_query(self, processed_folder):
        # Dummy implementation: returns a static query list.
        dummy_query = {
            "type": "template",
            "details": "dummy_template",
            "paths": [os.path.join(processed_folder, "dummy_template.png")],
            "confidence_threshold": 0.9,
            "overlap_threshold": 0.1,
            "has_text": True
        }
        return [dummy_query]

    def process_query(self, job_path=None):
        if not job_path or not os.path.exists(job_path):
            raise FileNotFoundError(f"Query file not found: {job_path}")
        with open(job_path, "r") as f:
            job = json.load(f)
        
        result = []

        debug_config = job.get("debug", {})
        preprocess_config = job.get("preprocess", {})
        query = job.get("query", [])

        if preprocess_config:
            print("Preprocess configuration found: performing preprocessing.")
            if self.__preprocessed_template == False:
                self.__folder_preprocessing(
                    preprocess_config.get("template_source_folder", None),
                    preprocess_config.get("template_processed_folder", None),
                    preprocess_function=self.__image_preprocessing_template,
                    scaling_factor=preprocess_config.get("scaling_factor", 1),
                    threshold_val=preprocess_config.get("binary_threshold_val", 160)
                )
                self.__preprocessed_template = True
            self.__folder_preprocessing(
                preprocess_config.get("target_source_folder", None),
                preprocess_config.get("target_processed_template_folder", None),
                preprocess_function=self.__image_preprocessing_template,
                scaling_factor=preprocess_config.get("scaling_factor", 1),
                threshold_val=preprocess_config.get("binary_threshold_val", 160)
            )
            self.__folder_preprocessing(
                preprocess_config.get("target_source_folder", None),
                preprocess_config.get("target_processed_text_folder", None),
                preprocess_function=self.__image_preprocessing_text
            )
        else:
            print("Preprocess configuration missing: skipping preprocessing.")

        if debug_config:
            print("Debug configuration active: enabling debug mode.")
            self.__visualize_template_match = debug_config.get("visualize_template_match", False)
            self.__visualize_ORB_matching = debug_config.get("visualize_ORB_matching", False)
            self.__visualize_text_recognition = debug_config.get("visualize_text_recognition", False)
        else:
            print("Debug configuration not active: using standard settings.")

        if query:
            print("Queries found: processing queries.")
            query.sort(key=lambda q: q.get("layout_priority", 9))
            source_path = os.path.join(
                preprocess_config.get("target_processed_template_folder", None),
                "source.png"
            )
            
            # FIRST FIND ALL LAYOUTS
            for layout_question in query:
                layout_question["response"] = self.__find_layout(
                    source_path=source_path,
                    question=layout_question
                )

            layout_priority_data = []
            for q in query:
                layout_priority = q.get("layout_priority")
                for response_item in q.get("response", []):
                    layout_priority_data.append({
                        "response": response_item,
                        "layout_priority": layout_priority
                    })

            # Group layout_priority_data by the "layout_priority" key

            layout_priority_data.sort(key=lambda item: item["layout_priority"])
            grouped_data = {
                key: list(group)
                for key, group in groupby(layout_priority_data, key=lambda item: item["layout_priority"])
            }

            #TODO: improve, and especially how do I then associate the results? a bit meh
            for priority, items in grouped_data.items():
                merged_entities = []
                for item in items:
                    merged_entities.extend(item["response"])
                # Assume that all items in the group share the same threshold values
                confidence_threshold = items[0].get("confidence_threshold", 0.9)
                overlap_threshold = items[0].get("overlap_threshold", 0.1)
                self.__remove_duplicates(merged_entities, confidence_threshold, overlap_threshold)
            
                
            # THEN DO AN NMS TO ELIMINATE DUPLICATES WITH THE APPROPRIATE FUNCTION BASED ON PRIORITY
            # FINALLY SEARCH FOR TEXT IN WHAT REMAINS, IF THE FIND TEXT FIELD EXISTS
        else:
            print("No queries specified: skipping query processing.")

        return result
