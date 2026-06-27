import { loadUser } from "./service";

export function consume(userId: string) {
  return loadUser(userId);
}
